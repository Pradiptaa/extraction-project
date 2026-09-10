"""Builds the embedding view: one row per `raw_extraction.json` node, carrying
just what a future chunker/embedder needs (a durable id, its place in the
tree, and the text to embed) instead of the full audit-fidelity node shape.

Deliberately NOT the chunker. Node text is not split or merged here — every
row is a 1:1 projection of one tree node, so this stage's own correctness is
checkable by node-count parity with `raw_extraction.json` (see
`retrieval/tests/test_build_embedding_view.py`). Splitting long nodes into
token-sized chunks, and deciding whether short sibling nodes should merge,
belongs to a later stage that starts from this file's output.

    python -m retrieval.build_embedding_view path/to/raw_extraction.json --out retrieval_output
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from .schema import EMBEDDING_SCHEMA_VERSION, embedding_id

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def _node_text(node: dict) -> str:
    """Title and body concatenated, whitespace-collapsed. Both, not just one:
    a clause's `title` ("Cacat Mutu") is where its topic lives, `text_raw` is
    where the actual obligation text lives, and a similarity search needs
    both in the embedded string to match on either."""
    title = node.get("title") or ""
    text_raw = node.get("text_raw") or ""
    combined = f"{title}\n{text_raw}" if title and text_raw else (title or text_raw)
    return _WS_RE.sub(" ", combined).strip()


def build_embedding_view(document: dict) -> dict:
    """Returns the embedding-view document: schema_version, a pointer back to
    the source raw_extraction, and one row per node in `structure[]`."""
    source = document.get("source") or {}
    nodes = document.get("structure") or []

    # Identifies which contract a node came from, so the same clause text in
    # two contracts gets two ids. Falls back to the filename only if the
    # pipeline wrote no sha256; a missing key would silently merge documents.
    document_key = source.get("sha256") or source.get("file") or ""
    if not document_key:
        logger.warning("source has no sha256 or file — ids cannot be scoped per document")

    rows = []
    empty_text_count = 0
    seen: Counter[tuple] = Counter()
    for node in nodes:
        text = _node_text(node)
        if not text:
            empty_text_count += 1
        page = (node.get("pages") or [None])[0]
        path = node.get("path") or []
        label = node.get("label_normalized")
        sub_doc = node.get("sub_document")

        # Occurrence ordinal among nodes identical in every other component;
        # see `schema.embedding_id` for why this is the last-resort tiebreak.
        dedupe_key = (sub_doc or "", page, tuple(path), label or "", text)
        occurrence = seen[dedupe_key]
        seen[dedupe_key] += 1

        rows.append(
            {
                "embedding_id": embedding_id(document_key, sub_doc, page, path, label, text, occurrence),
                "document_key": document_key,
                "node_id": node.get("node_id"),
                "parent_id": node.get("parent_id"),
                "node_type": node.get("node_type"),
                "sub_document": sub_doc,
                "hierarchy_path": path,
                "label_normalized": label,
                "depth": node.get("depth"),
                "pages": node.get("pages") or [],
                "text": text,
            }
        )

    if empty_text_count:
        logger.warning("%d of %d nodes produced empty embedding text (no title and no text_raw)", empty_text_count, len(nodes))

    distinct_ids = len({r["embedding_id"] for r in rows})
    if distinct_ids != len(rows):
        # Not recoverable here: loading these into Chroma would silently drop
        # the duplicates, so fail loudly rather than emit a lossy view.
        raise ValueError(
            f"embedding_id is not unique: {len(rows)} nodes produced {distinct_ids} ids "
            f"({len(rows) - distinct_ids} collisions)"
        )

    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "source": {
            "raw_extraction_sha256": source.get("sha256"),
            "file": source.get("file"),
            "extracted_at": source.get("extracted_at"),
        },
        "node_count": len(rows),
        "nodes": rows,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build the embedding view from raw_extraction.json")
    parser.add_argument("raw_path", type=Path, help="Path to raw_extraction.json")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: alongside the input)")
    args = parser.parse_args()

    if not args.raw_path.exists():
        logger.error("%s does not exist", args.raw_path)
        return 1

    document = json.loads(args.raw_path.read_text(encoding="utf-8"))
    view = build_embedding_view(document)

    out_dir = args.out or args.raw_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    out_path = out_dir / f"{stem}_embedding_view.json"
    out_path.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("%d nodes -> %s", view["node_count"], out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
