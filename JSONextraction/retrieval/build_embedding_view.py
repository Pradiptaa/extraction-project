"""Builds the embedding view: one row per `raw_extraction.json` tree node and one
per ruled-table row, carrying only a durable id, its place in the document, and
the text to embed.

Not the chunker — every row is a 1:1 projection, so correctness is checkable by
row-count parity with `raw_extraction.json`. Splitting and merging belong to a
later stage that starts from this file's output.

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
    """Title and body concatenated, whitespace-collapsed. Both, since the title
    carries the topic and `text_raw` the obligation."""
    title = node.get("title") or ""
    text_raw = node.get("text_raw") or ""
    combined = f"{title}\n{text_raw}" if title and text_raw else (title or text_raw)
    return _WS_RE.sub(" ", combined).strip()


def build_embedding_view(document: dict) -> dict:
    """Returns the embedding-view document: schema_version, a pointer back to
    the source raw_extraction, and one row per node in `structure[]`."""
    source = document.get("source") or {}
    nodes = document.get("structure") or []

    # Scopes ids per contract, so the same clause text in two gets two ids.
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

        # Last-resort tiebreak; see `schema.embedding_id`.
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

    table_rows, skipped_empty = _table_rows(document, nodes, document_key, seen)
    if skipped_empty:
        logger.info("%d blank table rows skipped (no cell text)", skipped_empty)
    rows.extend(table_rows)

    distinct_ids = len({r["embedding_id"] for r in rows})
    if distinct_ids != len(rows):
        # Chroma would silently drop the duplicates, so fail loudly instead.
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
        # Split out so an extra table row can't mask a missing tree node.
        "node_count": len(rows),
        "structure_row_count": len(rows) - len(table_rows),
        "table_row_count": len(table_rows),
        "table_rows_skipped_empty": skipped_empty,
        "nodes": rows,
    }


def _cells_text(cells: list) -> str:
    return " | ".join(c for c in (_WS_RE.sub(" ", cell or "").strip() for cell in cells) if c)


def _sub_document_by_page(nodes: list[dict]) -> dict[int, str | None]:
    """Each page's sub-document, carried forward across pages with no tree nodes
    (ruled-table pages usually have none)."""
    counts: dict[int, Counter] = {}
    for node in nodes:
        for page in node.get("pages") or []:
            counts.setdefault(page, Counter())[node.get("sub_document")] += 1
    if not counts:
        return {}
    result: dict[int, str | None] = {}
    current = None
    for page in range(1, max(counts) + 1):
        if page in counts:
            current = counts[page].most_common(1)[0][0]
        result[page] = current
    return result


def _table_rows(document: dict, nodes: list[dict], document_key: str, seen: Counter) -> tuple[list[dict], int]:
    """One embedding row per ruled-table row, from `tables[]`.

    Text is the non-empty cells joined by " | ", without column headers, which
    would make every row of a sheet look alike to a retriever.

    `headers` is not always a header: on a continuation page pdfplumber takes a
    data row as one, and it is not repeated in `rows`. So the header is emitted
    as its own row unless its text duplicates one already emitted — costing one
    vector, versus silently dropping contract data the other way.

    Cross-references are carried as `refs` so the gate can check that an SSKK row
    still points at its SSUK clause.
    """
    by_id = {n.get("node_id"): n for n in nodes}
    sub_by_page = _sub_document_by_page(nodes)
    rows: list[dict] = []
    headers_seen: set[str] = set()
    skipped_empty = 0

    for table in document.get("tables") or []:
        table_id = table.get("table_id") or ""
        page = table.get("page")
        sub_doc = sub_by_page.get(page)

        entries: list[tuple[str, list, list]] = []
        header_text = _cells_text(table.get("headers") or [])
        if header_text and header_text not in headers_seen:
            headers_seen.add(header_text)
            entries.append(("h", table.get("headers") or [], []))
        for index, row in enumerate(table.get("rows") or []):
            entries.append((str(index), row.get("cells") or [], row.get("refs_out") or []))

        for row_key, cells, refs_out in entries:
            text = _cells_text(cells)
            if not text:
                # A blank grid row still embeds to a real vector that can rank
                # near a query, so it is counted rather than emitted.
                skipped_empty += 1
                continue
            path = [table_id, row_key]
            refs = []
            for ref in refs_out:
                target = by_id.get(ref.get("resolved_node_id"))
                refs.append({
                    "raw": ref.get("raw"),
                    "type": ref.get("type"),
                    "target_sub_document": target.get("sub_document") if target else None,
                    "target_path": target.get("path") if target else None,
                })
            dedupe_key = (sub_doc or "", page, tuple(path), "", text)
            occurrence = seen[dedupe_key]
            seen[dedupe_key] += 1
            rows.append({
                "embedding_id": embedding_id(document_key, sub_doc, page, path, None, text, occurrence),
                "document_key": document_key,
                "node_id": None,
                "parent_id": None,
                "node_type": "table_row",
                "sub_document": sub_doc,
                "hierarchy_path": path,
                "label_normalized": None,
                "depth": None,
                "pages": [page] if page is not None else [],
                "text": text,
                "table_id": table_id,
                "refs": refs,
            })

    return rows, skipped_empty


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
