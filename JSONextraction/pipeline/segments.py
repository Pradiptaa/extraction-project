"""Where each sub-document begins — decided per block, not per page.

`main.assign_sub_documents` takes each marker's first matching *page* and
carries it forward. Three things break that:

* a contents page names every part before any of them starts, so the whole
  document inherits the last one listed — measured on the authored
  `parts_toc_and_caps` specimen, where all 12 nodes came out `special_terms`;
* a part that starts mid-page labels the whole page, including the tail of the
  part before it;
* a marker quoted in prose ("sebagaimana diatur dalam Syarat-Syarat Umum
  Kontrak") reads the same as a heading.

Boundaries are found at block granularity, a contents page is recognised and
skipped, and a marker only starts a part when the line it sits on reads as a
heading. Blocks before the first boundary are left unassigned (`None`), which
is what "we do not know yet" has always meant in this schema.
"""
from __future__ import annotations

import re

from .headings import PageStats, score_heading

# A contents line: "SYARAT-SYARAT UMUM KONTRAK .......... 12", or any line
# ending in a page number after a run of leaders or wide spacing.
_TOC_LINE_RE = re.compile(r"(\.{3,}|\s{4,}|·{3,}|_{3,})\s*\d{1,4}\s*$")
_TOC_TITLE_RE = re.compile(r"^(daftar\s+isi|table\s+of\s+contents)\b", re.IGNORECASE)
# A page is a contents page when this many of its lines look like entries.
TOC_LINE_SHARE = 0.4
TOC_MIN_LINES = 3


def is_toc_line(text: str) -> bool:
    return bool(_TOC_LINE_RE.search(text.strip()))


def find_toc_blocks(pages_blocks: dict[int, list], page_order: list[int]) -> set[tuple[int, int]]:
    """The blocks that make up a table of contents.

    Found as a *run* of contents lines rather than a whole page: a contract may
    print its contents above the first heading on the same page, and marking
    the page would then hide that heading too.
    """
    toc_blocks: set[tuple[int, int]] = set()
    for page in page_order:
        blocks = pages_blocks.get(page) or []
        run: list[int] = []
        announced = False
        for index, block in enumerate(blocks):
            text = block.text.strip()
            if _TOC_TITLE_RE.match(text):
                announced = True
                run = [index]
                continue
            if is_toc_line(text):
                run.append(index)
                continue
            # The run ends. Keep it when it is long enough to be a contents
            # list, or when the page announced one.
            if len(run) >= TOC_MIN_LINES or (announced and len(run) >= 2):
                toc_blocks.update((page, i) for i in run)
            run, announced = [], announced and False
        if len(run) >= TOC_MIN_LINES or (announced and len(run) >= 2):
            toc_blocks.update((page, i) for i in run)
    return toc_blocks


def assign_sub_documents_by_block(
    pages_blocks: dict[int, list],
    page_order: list[int],
    profile: dict,
    page_width_by_page: dict[int, float] | None = None,
    line_height_by_page: dict[int, float] | None = None,
) -> tuple[dict[tuple[int, int], str | None], list[str]]:
    """{(page, block index): sub-document}, plus notes for the quality block."""
    markers = [(m["name"], re.compile(m["start"], re.MULTILINE)) for m in profile.get("sub_document_markers") or []]
    page_width_by_page = page_width_by_page or {}
    line_height_by_page = line_height_by_page or {}
    notes: list[str] = []
    if not markers:
        return {}, notes

    toc_blocks = find_toc_blocks(pages_blocks, page_order)
    if toc_blocks:
        notes.append(f"toc_blocks_skipped={len(toc_blocks)}")

    assignment: dict[tuple[int, int], str | None] = {}
    seen: set[str] = set()
    current: str | None = None

    for page in page_order:
        blocks = pages_blocks.get(page) or []
        stats = PageStats.from_blocks(blocks, page_width_by_page.get(page, 595.0),
                                      line_height_by_page.get(page, 12.0))
        previous_bottom = None
        previous_text = None
        for index, block in enumerate(blocks):
            text = block.text.strip()
            if (page, index) not in toc_blocks and not is_toc_line(text):
                for name, pattern in markers:
                    if name in seen or not pattern.search(text):
                        continue
                    # The marker must sit on a line that reads as a heading, so
                    # the same words quoted inside a sentence do not start a part.
                    score = score_heading(
                        text, stats=stats, x0=block.x0, x1=block.x1,
                        font_size=block.font_size, is_bold=block.is_bold,
                        gap_above=None if previous_bottom is None else block.top - previous_bottom,
                        previous_text=previous_text,
                    )
                    if not score.is_heading:
                        notes.append(f"marker_not_a_heading page={page} name={name} score={score.score:.2f}")
                        continue
                    seen.add(name)
                    current = name
                    break
            assignment[(page, index)] = current
            previous_bottom, previous_text = block.bottom, block.text

    missing = [name for name, _ in markers if name not in seen]
    if missing:
        notes.append(f"markers_not_found={missing}")
    return assignment, notes


def page_level_view(assignment: dict[tuple[int, int], str | None], page_order: list[int]) -> dict[int, str | None]:
    """The page's own label: the part in effect where the page ends.

    `pages[]` keeps a single value per page for compatibility, while nodes carry
    the block-level answer. A page where a new part begins belongs to that new
    part — it owns the rest of the page and everything after — which is also
    what the page-level assignment this replaces meant by "from here onward".

    Not a majority vote: on a boundary page most surviving blocks belong to the
    part that is *ending*, and a ruled-table page keeps only the few blocks
    outside its tables. That made the SSKK's own page report `general_terms`,
    and every table row on it inherited the wrong part.
    """
    out: dict[int, str | None] = {}
    carried: str | None = None
    for page in page_order:
        values = [v for (p, _), v in sorted(assignment.items()) if p == page]
        for value in values:
            if value is not None:
                carried = value
        out[page] = carried if values else carried
    return out
