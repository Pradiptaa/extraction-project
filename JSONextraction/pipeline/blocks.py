"""Stage 5 — BLOCK EXTRACTION. Layout-appropriate extractor emits ordered flat
text blocks (line granularity) with bbox + font attributes. No hierarchy yet —
that is Stage 6 (tree.py). Ruled tables are extracted separately into cell
grids via pdfplumber, which is what avoids the SSKK cell-wrap column bleed
that flat `-layout` text produces (see analisis_pipeline_kontrak.md A.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pdfplumber

from .layout import LayoutInfo, _line_groups
from .probe import PageProbe


@dataclass
class TextBlock:
    page: int
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    font_size: float
    is_bold: bool
    column_index: int      # 0 = left/only column, 1 = right column (two_column only)


@dataclass
class TableBlock:
    page: int
    rows: list[list[str]]
    bbox: dict
    extraction_method: str = "pdfplumber_ruled"


# Row bucket for sorting, in points. A clause heading (column 0) and its own
# first line of body/subclause text (column 1) are computed as separate
# blocks from the same merged line, so their `top` values can differ by a
# fraction of a point — but `round(x, 0)` uses banker's-rounding, which can
# put two values as close as 504.5/504.6 into DIFFERENT integer buckets when
# they straddle a .5 tie. That flips their order relative to other rows,
# which cascades into a subclause attaching to the wrong parent clause. A
# coarser bucket (well under the ~12-14pt line height in this document, so
# genuinely different rows never merge) makes the tie-break moot.
_ROW_BUCKET_PT = 6.0


def _row_bucket(top: float) -> int:
    return round(top / _ROW_BUCKET_PT)


def _line_to_block(line: list[dict], page: int, column_index: int) -> TextBlock:
    line_sorted = sorted(line, key=lambda w: w["x0"])
    text = " ".join(w["text"] for w in line_sorted)
    x0 = min(w["x0"] for w in line_sorted)
    x1 = max(w["x1"] for w in line_sorted)
    top = min(w["top"] for w in line_sorted)
    bottom = max(w["bottom"] for w in line_sorted)
    font_size = line_sorted[0].get("size", 0.0)
    is_bold = any("bold" in (w.get("fontname") or "").lower() for w in line_sorted)
    return TextBlock(page, text, x0, top, x1, bottom, font_size, is_bold, column_index)


def _split_merged_line(line_sorted: list[dict], right_start_x: float, margin: float = 6.0) -> int | None:
    """Finds where a line that merges left-column and right-column words
    should split. Uses the right column's own (tightly consistent) start
    position, not the page's midpoint boundary: a long left-column heading
    can run well past the midpoint before wrapping, so a word's position
    relative to the midpoint is ambiguous right where it matters, while the
    body column starts at nearly the same x on every line. Returns the
    split index, or None if the line doesn't actually merge both columns."""
    threshold = right_start_x - margin
    split_idx = next((i for i, w in enumerate(line_sorted) if w["x0"] >= threshold), None)
    if split_idx is None or split_idx == 0:
        return None
    return split_idx


def extract_text_blocks(probe: PageProbe, layout: LayoutInfo) -> list[TextBlock]:
    lines = _line_groups(probe.words)
    if not lines:
        return []

    if layout.layout_type == "two_column" and layout.column_boundary_frac is not None:
        right_start_x = (layout.right_column_start_frac or layout.column_boundary_frac) * probe.width
        # A clause's left-column heading and its right-column body routinely
        # start at the same `top` (a short one-line heading beside the first
        # line of body text), so a single global line-grouping pass merges
        # their words into one line before column identity is ever assigned.
        # Split each such line at the right column's start position — this
        # is what lets the tree builder route a short heading to `title`
        # without also swallowing the body line beside it.
        blocks = []
        for line in lines:
            line_sorted = sorted(line, key=lambda w: w["x0"])
            split_idx = _split_merged_line(line_sorted, right_start_x)
            if split_idx is not None:
                blocks.append(_line_to_block(line_sorted[:split_idx], probe.page, 0))
                blocks.append(_line_to_block(line_sorted[split_idx:], probe.page, 1))
            else:
                column_index = 0 if line_sorted[0]["x0"] < right_start_x else 1
                blocks.append(_line_to_block(line_sorted, probe.page, column_index))
        # Row-major reading order: sort by top first, then column, then x0.
        # This is the coordinate-based fix for the reading-order collapse
        # that naive extraction produces (analisis_pipeline_kontrak.md A.4).
        blocks.sort(key=lambda b: (_row_bucket(b.top), b.column_index, b.x0))
        return blocks

    # single_column / form / mixed: plain top-to-bottom, left-to-right order.
    blocks = [_line_to_block(line, probe.page, 0) for line in lines]
    blocks.sort(key=lambda b: (_row_bucket(b.top), b.x0))
    return blocks


def extract_table_blocks(pdf_path: str, page_numbers: list[int]) -> dict[int, list[TableBlock]]:
    """Ruled-table extraction via pdfplumber's cell reconstruction, for pages
    already classified as `ruled_table`. Never flat `-layout` text for these."""
    result: dict[int, list[TableBlock]] = {}
    if not page_numbers:
        return result
    wanted = set(page_numbers)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if i not in wanted:
                continue
            tables = page.find_tables()
            page_tables = []
            for t in tables:
                rows = t.extract()
                clean_rows = [[(c or "").strip() for c in row] for row in rows]
                x0, top, x1, bottom = t.bbox
                page_tables.append(
                    TableBlock(
                        page=i,
                        rows=clean_rows,
                        bbox={"x0": x0, "top": top, "x1": x1, "bottom": bottom},
                    )
                )
            if page_tables:
                result[i] = page_tables
    return result
