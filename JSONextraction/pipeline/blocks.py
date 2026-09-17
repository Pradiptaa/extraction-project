"""Stage 5 — Block Extraction. Emits ordered flat text blocks; hierarchy is Stage 6."""
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


# Coarse enough that sub-point `top` differences within one row never split
# across buckets, but well under line height so real rows never merge.
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
    """Split index for a line merging both columns, or None. Keyed on the right
    column's own start position, not the page midpoint, which is ambiguous."""
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
        # Line grouping merges a heading and the body line beside it, so split
        # each such line back apart at the right column's start.
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
        # Row-major reading order.
        blocks.sort(key=lambda b: (_row_bucket(b.top), b.column_index, b.x0))
        return blocks

    blocks = [_line_to_block(line, probe.page, 0) for line in lines]
    blocks.sort(key=lambda b: (_row_bucket(b.top), b.x0))
    return blocks


def extract_table_blocks(pdf_path: str, page_numbers: list[int]) -> dict[int, list[TableBlock]]:
    """Cell-grid extraction for pages classified as `ruled_table`."""
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
