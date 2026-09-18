"""Rebuilds a collection's HNSW index from vectors that are already stored.

Chroma fixes HNSW parameters at creation, so changing them means a new index.
The vectors are unaffected, so they are copied across — no embedding calls, no
tokens spent. The source is left untouched, so a bad rebuild is reverted by
pointing back at the old collection name.

    python -m retrieval.reindex --from contracts__mistral-embed__v2_0_0
    python -m retrieval.reindex --from <old> --dry-run

`--verify` compares the new index against an exact brute-force scan and reports
recall by distance.
"""
from __future__ import annotations

import argparse
import logging
import sys

import chromadb
import numpy as np

from .config import INDEX_METADATA, load_settings

logger = logging.getLogger("retrieval.reindex")

BATCH = 500


def copy_collection(client, source_name: str, target_name: str, dry_run: bool = False):
    source = client.get_collection(source_name)
    total = source.count()
    logger.info("source     : %s (%d rows)", source_name, total)
    logger.info("target     : %s", target_name)
    logger.info("index      : %s", INDEX_METADATA)

    if not total:
        raise SystemExit(f"source collection {source_name!r} is empty — nothing to reindex")

    existing = [c.name for c in client.list_collections()]
    if target_name in existing:
        target = client.get_collection(target_name)
        if target.count() == total:
            logger.info("target already holds %d rows — nothing to do", total)
            return target
        logger.warning(
            "target exists with %d of %d rows; re-copying (upsert is idempotent on id)",
            target.count(), total,
        )
    if dry_run:
        logger.info("dry run — would copy %d rows, 0 embedding calls", total)
        return None

    target = client.get_or_create_collection(target_name, metadata=INDEX_METADATA)

    copied = 0
    while copied < total:
        got = source.get(
            limit=BATCH, offset=copied, include=["documents", "metadatas", "embeddings"]
        )
        if not got["ids"]:
            break
        target.upsert(
            ids=got["ids"],
            embeddings=got["embeddings"],
            documents=got["documents"],
            metadatas=got["metadatas"],
        )
        copied += len(got["ids"])
        logger.info("copied %d/%d", copied, total)

    if target.count() != total:
        raise SystemExit(
            f"copied {target.count()} rows but source has {total} — refusing to report success"
        )
    logger.info("done: %d rows, 0 tokens spent", target.count())
    return target


def verify(collection, sample: int = 16, k: int = 5) -> float:
    """Recall@k by distance, against an exact scan of the collection's own
    vectors. By distance, not id: most of this corpus is duplicate text, so ids
    would mostly measure arbitrary tie-breaking."""
    got = collection.get(include=["embeddings"])
    vectors = np.asarray(got["embeddings"], dtype=np.float32)
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    rng = np.random.default_rng(0)
    probes = rng.choice(len(vectors), size=min(sample, len(vectors)), replace=False)

    total = 0.0
    for probe in probes:
        v = norms[probe]
        sims = norms @ v
        worst_true = 1 - np.sort(sims)[-k:][0]
        got_dist = collection.query(
            query_embeddings=[vectors[probe].tolist()], n_results=k, include=["distances"]
        )["distances"][0]
        total += sum(1 for d in got_dist if d <= worst_true + 1e-4) / k

    recall = total / len(probes)
    logger.info("recall@%d by distance over %d probes: %.3f", k, len(probes), recall)
    if recall < 0.999:
        logger.warning("index is losing true neighbours — check INDEX_METADATA")
    return recall


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Rebuild a collection's HNSW index, no re-embedding")
    parser.add_argument("--from", dest="source", required=True, help="Existing collection to copy from")
    parser.add_argument("--to", dest="target", default=None, help="Target (default: the configured collection)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Check recall by distance after copying")
    args = parser.parse_args()

    settings = load_settings()
    target_name = args.target or settings.collection
    if target_name == args.source:
        raise SystemExit(
            "source and target are the same collection. HNSW parameters cannot be changed in "
            "place — bump config.INDEX_TAG so the rebuild gets its own name."
        )

    client = chromadb.PersistentClient(path=str(settings.db_path))
    target = copy_collection(client, args.source, target_name, args.dry_run)
    if target is not None and args.verify:
        verify(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
