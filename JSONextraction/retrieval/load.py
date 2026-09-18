"""Embeds every row of one or more embedding views and loads them into Chroma.

Resumable: a run that dies partway must not re-spend tokens on rows already
done. Chroma alone decides what counts as embedded — a row is done if the
collection holds its id. The JSONL manifest is an audit log, not the source of
truth, so a manifest that lists rows Chroma lacks is logged as drift and those
rows are embedded again.

A transiently-failing batch is logged and skipped so one bad minute doesn't
waste the run; anything else (bad key, dimension change, Chroma write error)
aborts, since every later batch would fail the same way.

    python -m retrieval.load output/embedding/*.json
    python -m retrieval.load output/embedding/polres_embedding_view.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chromadb

from .config import INDEX_METADATA, Settings, load_settings
from .embed import Embedder, is_retryable
from .schema import EMBEDDING_SCHEMA_VERSION

logger = logging.getLogger("retrieval.load")


def _manifest_path(settings: Settings) -> Path:
    # Scoped by collection: a different model or schema version is another job.
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
                # Expected after a hard kill; earlier lines are still valid.
                logger.warning("manifest line %d is corrupt, ignoring it", line_no)
    return done


def append_manifest(path: Path, ids: list[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ids": ids}) + "\n")
        f.flush()


def _metadata(row: dict) -> dict:
    """Chroma metadata values must be scalars, so lists are flattened."""
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
        # Table rows only. Each ref flattens to "sub_document:path", or "?:raw"
        # when unresolved, so it stays visible rather than disappearing.
        "table_id": row.get("table_id") or "",
        "ref_targets": ";".join(
            f"{ref['target_sub_document']}:{'/'.join(ref['target_path'])}"
            if ref.get("target_path") else f"?:{ref.get('raw')}"
            for ref in row.get("refs") or []
            if ref.get("type") == "internal"
        ),
    }


def read_views(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        view = json.loads(path.read_text(encoding="utf-8"))
        version = view.get("schema_version")
        if version != EMBEDDING_SCHEMA_VERSION:
            raise SystemExit(
                f"{path.name} was built by schema {version}, this loader expects "
                f"{EMBEDDING_SCHEMA_VERSION}. Rebuild the view rather than mixing schemas."
            )
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
    """Copy vectors for pending rows out of an existing collection, by id.
    Returns the rows still pending afterwards.

    Sound because `embedding_id` hashes the row's text, so a matching id
    addresses byte-identical text; the stored document is compared anyway and a
    mismatch refused. Only ids in the new views are copied, so stale rows are
    left behind. The source must come from the same embedding model.
    """
    if f"__{settings.model}__" not in source_name:
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
        # Nothing written, so report what a real run would still embed.
        available = set()
        for start in range(0, len(ids), REUSE_BATCH):
            available |= set(source.get(ids=ids[start:start + REUSE_BATCH], include=[])["ids"])
        return [r for r in pending if r["embedding_id"] not in available]
    stored = set(collection.get(ids=ids, include=[])["ids"])
    return [r for r in pending if r["embedding_id"] not in stored]


def run(paths: list[Path], settings: Settings, dry_run: bool = False, reuse_from: str | None = None) -> int:
    rows = read_views(paths)

    settings.db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.db_path))
    collection = client.get_or_create_collection(settings.collection, metadata=INDEX_METADATA)

    manifest = _manifest_path(settings)
    # Chroma is the source of truth: a row is done only if it is stored.
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
        return 0

    chars = sum(len(r["text"]) for r in pending)
    batches = -(-len(pending) // settings.batch_size)
    logger.info(
        "will send %d requests (batch=%d), ~%d chars, ~%d tokens estimated",
        batches, settings.batch_size, chars, int(chars / 2.52),
    )
    if dry_run:
        logger.info("dry run — no API calls made, nothing written")
        return 0

    embedder = Embedder(settings.api_key, settings.model, settings.request_delay)
    failed_batches = 0

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
            # Only after the upsert returns; a retry is safe, upsert is idempotent.
            append_manifest(manifest, ids)
            logger.info("batch %d/%d ok (%d rows, %d tokens total)", index + 1, batches, len(chunk), embedder.total_tokens)
        except Exception as exc:
            failed_batches += 1
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
                break

    logger.info(
        "done: %d rows in collection, %d tokens used this run, %d batches failed",
        collection.count(), embedder.total_tokens, failed_batches,
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
    args = parser.parse_args()

    missing = [p for p in args.views if not p.exists()]
    if missing:
        logger.error("no such file(s): %s", ", ".join(str(p) for p in missing))
        return 1

    return run(args.views, load_settings(), dry_run=args.dry_run, reuse_from=args.reuse_from)


if __name__ == "__main__":
    sys.exit(main())
