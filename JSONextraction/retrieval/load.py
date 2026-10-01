from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from .config import INDEX_METADATA, Settings, load_settings
from .embed import Embedder, is_retryable
from .registry import connect as registry_connect, rebuild as registry_rebuild, registry_path
from .schema import EMBEDDING_SCHEMA_VERSION

logger = logging.getLogger("retrieval.load")


def _manifest_path(settings: Settings) -> Path:
    return settings.db_path / f"{settings.collection}.manifest.jsonl"


def load_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                done.update(json.loads(line)["ids"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("manifest line %d is corrupt, ignoring it", line_no)
    return done


def append_manifest(path: Path, ids: list[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ids": ids}) + "\n")
        f.flush()


def _metadata(row: dict) -> dict:
    pages = row.get("pages") or []
    return {
        "document_key": row.get("document_key") or "",
        "node_id": row.get("node_id") or "",
        "node_type": row.get("node_type") or "",
        "sub_document": row.get("sub_document") or "",
        "label": row.get("label_normalized") or "",
        "hierarchy_path": "/".join(row.get("hierarchy_path") or []),
        "depth": row.get("depth") if isinstance(row.get("depth"), int) else -1,
        "page_first": pages[0] if pages else -1,
        "page_last": pages[-1] if pages else -1,
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "table_id": row.get("table_id") or "",
        "tree_engine": row.get("tree_engine") or "",
        "ref_targets": ";".join(
            f"{ref['target_sub_document']}:{'/'.join(ref['target_path'])}"
            if ref.get("target_path") else f"?:{ref.get('raw')}"
            for ref in row.get("refs") or []
            if ref.get("type") == "internal"
        ),
    }


def views_engine(engines: dict[str, str | None]) -> str | None:
    known = {e for e in engines.values() if e}
    if len(known) > 1:
        raise SystemExit(
            "the given views were extracted with different tree engines "
            f"({', '.join(f'{name}={e}' for name, e in sorted(engines.items()) if e)}) — "
            "re-extract them with one engine"
        )
    return known.pop() if known else None


def check_collection_engine(collection, engine: str | None) -> None:
    stamped = (collection.metadata or {}).get("tree_engine")
    if stamped and engine and stamped != engine:
        raise SystemExit(
            f"collection {collection.name!r} holds {stamped!r} extractions but these views are {engine!r} — "
            "load them into a collection built with the same engine"
        )
    if not stamped:
        logger.warning("collection %s has no tree_engine stamp (created before B1)", collection.name)


def read_views(paths: list[Path], allow_failed: bool = False,
               engines: dict[str, str | None] | None = None) -> list[dict]:
    rows: list[dict] = []
    engines = {} if engines is None else engines
    for path in paths:
        view = json.loads(path.read_text(encoding="utf-8"))
        version = view.get("schema_version")
        if version != EMBEDDING_SCHEMA_VERSION:
            raise SystemExit(
                f"{path.name} was built by schema {version}, this loader expects "
                f"{EMBEDDING_SCHEMA_VERSION}. Rebuild the view rather than mixing schemas."
            )
        status = (view.get("source") or {}).get("pipeline_status")
        if status is None:
            logger.warning("%s has no pipeline_status (built before A3) — loading it unchecked", path.name)
        elif status != "passed":
            if not allow_failed:
                raise SystemExit(
                    f"{path.name} comes from an extraction with pipeline_status={status!r}. "
                    "Fix the extraction, or pass --allow-failed to load it anyway."
                )
            logger.warning("%s has pipeline_status=%r — loading it because of --allow-failed", path.name, status)
        engine = (view.get("source") or {}).get("tree_engine")
        if engine is None:
            logger.warning("%s has no tree_engine (built before B1) — its engine cannot be checked", path.name)
        engines[path.name] = engine
        for node in view["nodes"]:
            node["tree_engine"] = engine
        rows.extend(view["nodes"])
        logger.info("%s: %d nodes", path.name, view["node_count"])

    ids = [r["embedding_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(
            f"{len(ids) - len(set(ids))} duplicate embedding_ids across the given views — "
            "loading these would silently drop rows. Rebuild the views."
        )
    return rows


REUSE_BATCH = 500


def reuse_vectors(client, source_name: str, collection, pending: list[dict], settings: Settings,
                  manifest: Path, dry_run: bool) -> list[dict]:
    if f"__{settings.model_slug}__" not in source_name:
        raise SystemExit(
            f"--reuse-from {source_name!r} was not built by {settings.model!r} — refusing to mix "
            "vectors from different models"
        )
    try:
        source = client.get_collection(source_name)
    except Exception:
        raise SystemExit(f"--reuse-from collection {source_name!r} does not exist")

    by_id = {r["embedding_id"]: r for r in pending}
    ids = list(by_id)
    reused = 0
    for start in range(0, len(ids), REUSE_BATCH):
        got = source.get(ids=ids[start:start + REUSE_BATCH], include=["embeddings", "documents"])
        if not got["ids"]:
            continue
        mismatched = [i for i, doc in zip(got["ids"], got["documents"]) if doc != by_id[i]["text"]]
        if mismatched:
            raise SystemExit(
                f"{len(mismatched)} ids in {source_name!r} hold different text than the view "
                f"(first: {mismatched[0]}) — the id derivation is not what this loader assumes"
            )
        reused += len(got["ids"])
        if dry_run:
            continue
        chunk = [by_id[i] for i in got["ids"]]
        collection.upsert(
            ids=got["ids"],
            embeddings=got["embeddings"],
            documents=[r["text"] for r in chunk],
            metadatas=[_metadata(r) for r in chunk],
        )
        append_manifest(manifest, got["ids"])

    logger.info("reuse from %s: %d of %d pending rows %s, 0 tokens", source_name, reused, len(pending),
                "available" if dry_run else "copied")
    if dry_run:
        available = set()
        for start in range(0, len(ids), REUSE_BATCH):
            available |= set(source.get(ids=ids[start:start + REUSE_BATCH], include=[])["ids"])
        return [r for r in pending if r["embedding_id"] not in available]
    stored = set(collection.get(ids=ids, include=[])["ids"])
    return [r for r in pending if r["embedding_id"] not in stored]


def collection_row_counts(collection, keys: set[str]) -> dict[str, int]:
    return {
        key: len(collection.get(where={"document_key": key}, include=[])["ids"])
        for key in keys
    }


def register_documents(settings: Settings, collection, rows: list[dict],
                       stored_ids: set[str]) -> int:
    keys = {
        row["document_key"]
        for row in rows
        if row.get("document_key") and row["embedding_id"] in stored_ids
    }
    if not keys:
        return 0
    try:
        counts = {key: n for key, n in collection_row_counts(collection, keys).items() if n}
        connection = registry_connect(registry_path(settings.db_path))
        registered = registry_rebuild(
            connection, settings.collection,
            only_keys=set(counts), row_counts=counts,
            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        connection.close()
        return registered
    except Exception as exc:
        logger.warning(
            "could not update the document registry (%s: %s) — the collection is unaffected; "
            "run `python -m retrieval.registry rebuild` to refresh it",
            type(exc).__name__, exc,
        )
        return 0


def run(paths: list[Path], settings: Settings, dry_run: bool = False, reuse_from: str | None = None,
        allow_failed: bool = False) -> int:
    engines: dict[str, str | None] = {}
    rows = read_views(paths, allow_failed, engines)
    engine = views_engine(engines)

    settings.db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.db_path))
    metadata = dict(INDEX_METADATA, tree_engine=engine) if engine else INDEX_METADATA
    collection = client.get_or_create_collection(settings.collection, metadata=metadata)
    check_collection_engine(collection, engine)

    manifest = _manifest_path(settings)
    done = set(collection.get(include=[])["ids"]) if collection.count() else set()
    recorded = load_manifest(manifest)
    wanted = {r["embedding_id"] for r in rows}
    drift = (recorded & wanted) - done
    if drift:
        logger.warning(
            "manifest records %d of these rows as loaded but the collection does not hold them "
            "(collection deleted or recreated?) — embedding them again", len(drift),
        )

    pending = [r for r in rows if r["embedding_id"] not in done]

    logger.info("collection : %s", settings.collection)
    logger.info("db path    : %s", settings.db_path)
    logger.info("total rows : %d | already done: %d | pending: %d", len(rows), len(rows) - len(pending), len(pending))

    if pending and reuse_from:
        if reuse_from == settings.collection:
            raise SystemExit("--reuse-from names the target collection itself")
        pending = reuse_vectors(client, reuse_from, collection, pending, settings, manifest, dry_run)
        logger.info("pending after reuse: %d", len(pending))

    if not pending:
        logger.info("nothing to do — every row is already embedded and loaded")
        if not dry_run:
            register_documents(settings, collection, rows, {r["embedding_id"] for r in rows})
        return 0

    chars = sum(len(r["text"]) for r in pending)
    batches = -(-len(pending) // settings.batch_size)
    logger.info(
        "will send %d requests (batch=%d), ~%d chars", batches, settings.batch_size, chars,
    )
    if dry_run:
        logger.info("dry run — nothing embedded, nothing written")
        return 0

    embedder = Embedder(settings.model, settings.host)
    failed_batches = 0
    failed_ids: set[str] = set()

    for index in range(batches):
        chunk = pending[index * settings.batch_size : (index + 1) * settings.batch_size]
        ids = [r["embedding_id"] for r in chunk]
        try:
            vectors = embedder.embed([r["text"] for r in chunk])
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=[r["text"] for r in chunk],
                metadatas=[_metadata(r) for r in chunk],
            )
            append_manifest(manifest, ids)
            logger.info("batch %d/%d ok (%d rows, %d tokens total)", index + 1, batches, len(chunk), embedder.total_tokens)
        except Exception as exc:
            failed_batches += 1
            failed_ids.update(ids)
            logger.error(
                "batch %d/%d FAILED: %s: %s | first_id=%s last_id=%s node_ids=%s documents=%s",
                index + 1, batches, type(exc).__name__, exc,
                ids[0], ids[-1],
                [r.get("node_id") for r in chunk[:5]],
                sorted({r.get("document_key", "")[:12] for r in chunk}),
            )
            if not is_retryable(exc):
                skipped = batches - index - 1
                logger.error(
                    "aborting: %s is not a transient failure, so the %d remaining batch(es) "
                    "would fail the same way. Fix the cause and re-run; completed rows are kept.",
                    type(exc).__name__, skipped,
                )
                failed_batches += skipped
                failed_ids.update(
                    r["embedding_id"] for r in pending[(index + 1) * settings.batch_size:]
                )
                break

    registered = register_documents(
        settings, collection, rows, {r["embedding_id"] for r in rows} - failed_ids
    )

    logger.info(
        "done: %d rows in collection, %d documents registered, %d tokens embedded this run, "
        "%d batches failed",
        collection.count(), registered, embedder.total_tokens, failed_batches,
    )
    if failed_batches:
        logger.error("re-run the same command to retry only the failed rows")
        return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Embed embedding views and load them into Chroma")
    parser.add_argument("views", nargs="+", type=Path, help="One or more *_embedding_view.json files")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be sent, make no API calls")
    parser.add_argument(
        "--reuse-from",
        default=None,
        help="Existing collection (same embedding model) to copy vectors from by id before "
             "embedding anything — e.g. after a schema bump that kept ids stable",
    )
    parser.add_argument("--allow-failed", action="store_true",
                        help="Load views whose extraction did not pass validation")
    args = parser.parse_args()

    missing = [p for p in args.views if not p.exists()]
    if missing:
        logger.error("no such file(s): %s", ", ".join(str(p) for p in missing))
        return 1

    return run(args.views, load_settings(), dry_run=args.dry_run, reuse_from=args.reuse_from,
               allow_failed=args.allow_failed)


if __name__ == "__main__":
    sys.exit(main())
