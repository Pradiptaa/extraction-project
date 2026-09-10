"""Embeds every row of one or more embedding views and loads them into Chroma.

Resumable by design. A run that dies at node 2500 of 4021 — a rate cap, a
dropped connection, a Ctrl+C — must not re-spend tokens on the 2500 already
done, which matters more on a free tier than a paid one.

Progress is tracked in an append-only JSONL manifest rather than inferred from
Chroma itself. Chroma is queried too, and the two are unioned, but the manifest
is what makes "already embedded" durable: a row is only recorded after its
upsert returns, so a crash between the API call and the write is retried rather
than silently skipped.

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

from .config import Settings, load_settings
from .embed import Embedder
from .schema import EMBEDDING_SCHEMA_VERSION

logger = logging.getLogger("retrieval.load")


def _manifest_path(settings: Settings) -> Path:
    # Scoped by collection: a different model or schema version is a different
    # job, and must not inherit another run's completion record.
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
                # A half-written final line is expected after a hard kill.
                # Everything before it is still valid, so warn and keep going
                # rather than discarding a whole run's progress.
                logger.warning("manifest line %d is corrupt, ignoring it", line_no)
    return done


def append_manifest(path: Path, ids: list[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ids": ids}) + "\n")
        f.flush()


def _metadata(row: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool, so lists are
    flattened. `pages` becomes first/last rather than a JSON blob because those
    are what a reader would actually filter on."""
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


def run(paths: list[Path], settings: Settings, dry_run: bool = False) -> int:
    rows = read_views(paths)

    settings.db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.db_path))
    collection = client.get_or_create_collection(
        settings.collection, metadata={"hnsw:space": "cosine"}
    )

    manifest = _manifest_path(settings)
    done = load_manifest(manifest)
    if collection.count():
        # Union with what Chroma already holds, so a manifest deleted by hand
        # doesn't cause a full re-embed of rows that are already stored.
        existing = collection.get(include=[])["ids"]
        done |= set(existing)

    pending = [r for r in rows if r["embedding_id"] not in done]

    logger.info("collection : %s", settings.collection)
    logger.info("db path    : %s", settings.db_path)
    logger.info("total rows : %d | already done: %d | pending: %d", len(rows), len(rows) - len(pending), len(pending))

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
            # Only after the upsert returns: a crash before this point retries
            # the batch, which is safe because upsert is idempotent on id.
            append_manifest(manifest, ids)
            logger.info("batch %d/%d ok (%d rows, %d tokens total)", index + 1, batches, len(chunk), embedder.total_tokens)
        except Exception as exc:
            failed_batches += 1
            # Structured enough to resume or debug without re-running: which
            # batch, which rows, and what the source documents were.
            logger.error(
                "batch %d/%d FAILED: %s: %s | first_id=%s last_id=%s node_ids=%s documents=%s",
                index + 1, batches, type(exc).__name__, exc,
                ids[0], ids[-1],
                [r.get("node_id") for r in chunk[:5]],
                sorted({r.get("document_key", "")[:12] for r in chunk}),
            )

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
    args = parser.parse_args()

    missing = [p for p in args.views if not p.exists()]
    if missing:
        logger.error("no such file(s): %s", ", ".join(str(p) for p in missing))
        return 1

    return run(args.views, load_settings(), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
