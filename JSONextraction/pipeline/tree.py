"""Stage 6 — NUMBERING DETECTION & TREE BUILD.

Builds the generic recursive node tree from the flat, ordered text blocks
emitted by blocks.py. Depth is inferred from (numbering style, indentation,
font weight) and self-corrected with a sibling-sequence check; a genuine
backtracking search (per the design doc's stretch goal) is out of scope for
v1 — a sequence break is flagged (`sequence_break`) for human review instead
of triggering re-parsing with an alternate depth hypothesis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .blocks import TextBlock
from .numbering import match_numbering, sibling_successor
from .schema import NodeIdGenerator, ReadingOrderCounter

TERMINAL_PUNCT = (".", "!", "?", ":", ";", "…")

# A line with no lowercase letters, reasonably long, and containing enough
# real letters (not just a row of digits/punctuation) reads as a title in
# this document family: surveyed across every page of the sample PDF, every
# such line was a letterhead, Pasal/BAB title, SSUK part heading, table
# caption, or specimen-form title — never ordinary prose.
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,./()\-]{9,}$")
_MIN_ALLCAPS_LETTERS = 8


def _is_allcaps_heading(text: str) -> bool:
    if not _ALLCAPS_HEADING_RE.match(text):
        return False
    return sum(1 for c in text if c.isalpha()) >= _MIN_ALLCAPS_LETTERS


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    depth: int
    node_type: str
    label: str | None
    label_normalized: str | None
    numbering_style: str | None
    title: str | None
    path: list[str] = field(default_factory=list)
    text_raw: str = ""
    pages: list[int] = field(default_factory=list)
    page_labels: list[str] = field(default_factory=list)
    spans_page_break: bool = False
    bbox: dict | None = None
    char_span: dict | None = None
    reading_order: int = 0
    extraction: dict = field(default_factory=dict)
    refs_out: list = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    sub_document: str | None = None   # assigned post-hoc from the matched profile's markers


def _classify(style: str, label: str, column_index: int, layout_type: str, is_ssuk_body_page: bool) -> tuple[str, int]:
    """Returns (node_type, depth)."""
    if style == "chapter_word":
        return "part", 0
    if style == "article_word":
        return "article", 0
    if style == "letter_upper":
        return "section", 0
    if style == "letter_dotted":
        return "section", 1
    if style == "decimal_plain":
        # "1." means two different things depending on where it is: a
        # top-level SSUK clause heading (with its own title, in the
        # two-column body) versus an ordinary flat numbered list item
        # elsewhere (Tembusan lists, SPMK/SPPBJ instructions, the PAKTA
        # specimen forms' checklists) — those have no title/body split at
        # all, and treating them as a heading-bearing "clause" wrongly
        # empties their entire content out of text_raw and into `title`.
        # `layout_type == "two_column"` alone under-covers this: a genuine
        # SSUK clause can land on a page the layout detector calls
        # single_column (no left-column heading detected on THAT specific
        # page — pure body continuation), so a handful of real clauses
        # (verified: 32, 34, 78) would be misclassified too. Page height is
        # the reliable signal instead — the SSUK body is uniformly on
        # 612x792 (US Letter) pages; every other section of this document
        # uses larger front-matter/annex page sizes (~936pt tall) — see
        # analisis_pipeline_kontrak.md A.6.
        return ("clause", 1) if is_ssuk_body_page else ("list_item", 1)
    if style == "decimal_dotted":
        dots = label.count(".")
        return "subclause", 1 + dots
    if style in ("latin_lower", "roman_lower"):
        return "list_item", 3
    if style in ("paren_digit", "paren_latin"):
        return "list_item", 4
    if style == "bullet":
        return "list_item", 5
    return "paragraph", 2


SSUK_BODY_PAGE_HEIGHT = 792.0
SSUK_BODY_PAGE_HEIGHT_TOLERANCE = 10.0


def build_tree(
    pages_blocks: dict[int, list[TextBlock]],
    layout_by_page: dict[int, str],
    page_order: list[int],
    page_height_by_page: dict[int, float] | None = None,
) -> tuple[list[Node], dict[int, str], list[str]]:
    """Returns (nodes, page_raw_text_by_page, quality_flags)."""
    page_height_by_page = page_height_by_page or {}
    id_gen = NodeIdGenerator()
    order_gen = ReadingOrderCounter()
    nodes: dict[str, Node] = {}
    stack: list[Node] = []  # open ancestors, shallow-to-deep
    quality_flags: list[str] = []
    page_raw_text: dict[int, str] = {}
    page_cursor: dict[int, int] = {}
    last_label_by_parent_style: dict[tuple[str | None, str], str] = {}
    root_order: list[str] = []
    # Tracks the most recently opened "clause" node so a wrapped left-column
    # heading's continuation lines (2nd/3rd line of a multi-line clause
    # title) can be routed to its title instead of falling through to the
    # generic continuation path, which would splice them into the BODY
    # mid-sentence — the deepest open node by then is usually the clause's
    # first subclause, not the clause itself.
    current_clause_id: str | None = None
    # Set right after creating a part/section/article node; consumed by the
    # very next block if (and only if) it's a standalone ALL-CAPS heading
    # line — attached as/appended to that node's title — otherwise cleared
    # unconditionally on every other block. Covers both an EMPTY title
    # ("Pasal 3" alone on its line, the next line fills it in) and a
    # PARTIAL one ("B. PELAKSANAAN, PENYELESAIAN," already has a title from
    # its own match — a two-column split puts the rest, "ADENDUM DAN
    # PEMUTUSAN KONTRAK", on a separate line/column that must extend it,
    # not replace it or detach into its own node).
    pending_heading_target_id: str | None = None

    def cursor_append(page: int, text: str) -> tuple[int, int]:
        buf = page_raw_text.get(page, "")
        start = page_cursor.get(page, 0)
        page_raw_text[page] = buf + text + "\n"
        end = start + len(text)
        page_cursor[page] = end + 1
        return start, end

    # Vertical gap (points) within which two consecutive orphan blocks on a
    # ruled_table page are still the same wrapped caption/footnote, not two
    # unrelated ones.
    RULED_TABLE_CAPTION_GAP = 20.0
    # Same idea, for consecutive standalone ALL-CAPS heading lines outside
    # ruled_table pages (e.g. a multi-line letterhead).
    STANDALONE_HEADING_MERGE_GAP = 20.0
    # A running header (e.g. "LAMPIRAN A SYARAT-SYARAT KHUSUS KONTRAK",
    # repeated verbatim at the same `top` on multiple pages) sits close
    # enough above the real caption below it to pass the gap check, but is a
    # structurally distinct element and must never absorb — or be absorbed
    # by — a neighboring block.
    _RUNNING_HEADER_RE = re.compile(r"^LAMPIRAN\s+[A-Z]\b")

    for page in page_order:
        layout_type = layout_by_page.get(page, "single_column")
        blocks = pages_blocks.get(page, [])
        page_height = page_height_by_page.get(page)
        is_ssuk_body_page = (
            page_height is not None
            and abs(page_height - SSUK_BODY_PAGE_HEIGHT) <= SSUK_BODY_PAGE_HEIGHT_TOLERANCE
        )

        if layout_type == "ruled_table":
            # These are the blocks OUTSIDE every detected table's bbox on a
            # table-dominated page (table cell text goes to tables[], not
            # here) — typically short, scattered captions and footnotes
            # sitting between distinct tables, not flowing prose. Folding
            # them through the normal stack/continuation machinery below
            # glues unrelated captions and footnotes from different tables
            # into one node, because nothing ever closes the open leaf until
            # a new numbering match appears — and these blocks often have
            # none. Each one becomes its own standalone node instead; only
            # a block within RULED_TABLE_CAPTION_GAP points of the previous
            # one's bottom is treated as its wrapped continuation.
            last_bottom: float | None = None
            last_node_id: str | None = None
            for block in blocks:
                start, end = cursor_append(page, block.text)
                match = match_numbering(block.text)
                is_running_header = match is None and bool(_RUNNING_HEADER_RE.match(block.text))
                is_wrap = (
                    match is None
                    and not is_running_header
                    and last_node_id is not None
                    and last_node_id in nodes
                    and nodes[last_node_id].node_type != "header"
                    and last_bottom is not None
                    and (block.top - last_bottom) <= RULED_TABLE_CAPTION_GAP
                )
                if is_wrap:
                    leaf = nodes[last_node_id]
                    leaf.text_raw = f"{leaf.text_raw} {block.text}".strip()
                    leaf.char_span["end"] = end
                    last_bottom = block.bottom
                    continue

                node_id = id_gen.next()
                if match is not None:
                    node_type, depth = _classify(match.style, match.label_normalized, block.column_index, layout_type, is_ssuk_body_page)
                    label, label_normalized, numbering_style = match.label, match.label_normalized, match.style
                    text_raw, path = match.remainder, [match.label]
                    detector = f"{layout_type}:{match.style}"
                elif is_running_header:
                    node_type, depth = "header", 0
                    label = label_normalized = numbering_style = None
                    text_raw, path = block.text, []
                    detector = "ruled_table_running_header"
                else:
                    node_type, depth = "caption", 0
                    label = label_normalized = numbering_style = None
                    text_raw, path = block.text, []
                    detector = "ruled_table_orphan_block"

                node = Node(
                    node_id=node_id,
                    parent_id=None,
                    depth=depth,
                    node_type=node_type,
                    label=label,
                    label_normalized=label_normalized,
                    numbering_style=numbering_style,
                    title=None,
                    path=path,
                    text_raw=text_raw,
                    pages=[page],
                    bbox={"page": page, "x0": block.x0, "top": block.top, "x1": block.x1, "bottom": block.bottom},
                    char_span={"page": page, "start": start, "end": end},
                    reading_order=order_gen.next(),
                    extraction={"method": "native", "confidence": 1.0, "detector": detector, "flags": []},
                )
                nodes[node_id] = node
                root_order.append(node_id)
                last_node_id = node_id
                last_bottom = block.bottom
            continue

        prev_block_had_terminal = True
        # A standalone ALL-CAPS heading followed immediately (small vertical
        # gap) by ANOTHER ALL-CAPS line is one multi-line heading, not two —
        # a 4-line government letterhead being the clearest case. Reset per
        # page: nothing on one page should merge into a heading on another.
        last_standalone_heading_id: str | None = None
        last_block_bottom: float | None = None

        for block in blocks:
            start, end = cursor_append(page, block.text)
            match = match_numbering(block.text)

            if match is not None:
                node_type, depth = _classify(match.style, match.label_normalized, block.column_index, layout_type, is_ssuk_body_page)
                if layout_type == "two_column" and block.column_index == 1 and match.style == "decimal_plain":
                    # right-column plain numbers are body sub-references, not new clauses
                    node_type, depth = "subclause", 2

                # pop stack until we find a shallower-or-equal open ancestor
                while stack and stack[-1].depth >= depth:
                    stack.pop()
                parent = stack[-1] if stack else None

                # sequence validation (non-blocking; flags only)
                key = (parent.node_id if parent else None, match.style)
                prev_label = last_label_by_parent_style.get(key)
                if prev_label is not None:
                    expected = sibling_successor(match.style, prev_label)
                    if expected is not None and expected != match.label_normalized:
                        quality_flags.append(
                            f"sequence_break page={page} style={match.style} "
                            f"expected={expected} got={match.label_normalized}"
                        )
                last_label_by_parent_style[key] = match.label_normalized

                node_id = id_gen.next()
                path = (parent.path if parent else []) + [match.label]
                # For a heading-bearing type, match.remainder is the first
                # line of its TITLE, not body content — it goes to `title`
                # only. Leaving a copy in text_raw too meant later lines that
                # extend the (possibly wrapped) title only ever updated
                # `title`, leaving text_raw a stale first-line fragment that
                # both duplicated and truncated the real title.
                is_heading_type = node_type in ("part", "section", "clause") and bool(match.remainder)
                node = Node(
                    node_id=node_id,
                    parent_id=parent.node_id if parent else None,
                    depth=depth,
                    node_type=node_type,
                    label=match.label,
                    label_normalized=match.label_normalized,
                    numbering_style=match.style,
                    title=match.remainder if is_heading_type else None,
                    path=path,
                    text_raw="" if is_heading_type else match.remainder,
                    pages=[page],
                    bbox={"page": page, "x0": block.x0, "top": block.top, "x1": block.x1, "bottom": block.bottom},
                    char_span={"page": page, "start": start, "end": end},
                    reading_order=order_gen.next(),
                    extraction={
                        "method": "native",
                        "confidence": 1.0,
                        "detector": f"{layout_type}:{match.style}",
                        "flags": [],
                    },
                )
                nodes[node_id] = node
                if parent:
                    parent.children.append(node_id)
                else:
                    root_order.append(node_id)
                stack.append(node)
                if node_type == "clause":
                    current_clause_id = node_id
                # "Pasal 3" is followed on its own line by an ALL-CAPS title
                # ("HARGA KONTRAK, SUMBER PEMBIAYAAN DAN PEMBAYARAN") with no
                # numbering of its own; "B. PELAKSANAAN, PENYELESAIAN," (a
                # section) already has a partial title from its own match,
                # split across the two-column boundary, with the rest
                # ("ADENDUM DAN PEMUTUSAN KONTRAK") arriving as a separate
                # line to append rather than replace. Remember this node
                # either way so that next line extends its title (below)
                # instead of bleeding into its body or detaching entirely.
                pending_heading_target_id = node_id if node_type in ("part", "section", "article") else None
                last_standalone_heading_id = None
                last_block_bottom = block.bottom
                prev_block_had_terminal = node.text_raw.rstrip().endswith(TERMINAL_PUNCT) if node.text_raw else False
                continue

            # A clause's own wrapped left-column heading (handled just below)
            # takes priority over the general ALL-CAPS rule — a clause title
            # that happens to contain an all-caps fragment must still route
            # to the clause, not be treated as an unrelated document title.
            if (
                layout_type == "two_column"
                and block.column_index == 0
                and current_clause_id is not None
                and current_clause_id in nodes
            ):
                # A column-0 line with no numbering match, while a clause is
                # open, is a continuation of that clause's (wrapped)
                # left-column heading — never body text, which lives in
                # column 1. Route it to the clause's title, not text_raw.
                clause_node = nodes[current_clause_id]
                clause_node.title = f"{clause_node.title} {block.text}".strip() if clause_node.title else block.text
                pending_heading_target_id = None
                last_standalone_heading_id = None
                last_block_bottom = block.bottom
                prev_block_had_terminal = block.text.rstrip().endswith(TERMINAL_PUNCT)
                continue

            # A standalone ALL-CAPS line with no numbering of its own is,
            # everywhere it occurs in this document (surveyed across all 74
            # pages: letterhead, Pasal titles, SSUK part headings, table
            # captions, and specimen-form titles like "PAKTA KOMITMEN
            # KESELAMATAN KONSTRUKSI"), a title — never ordinary prose.
            # Three cases, checked in order: (1) right after an empty/partial
            # article/part/section title it extends that node's title; (2)
            # immediately after another standalone ALL-CAPS heading (small
            # vertical gap) it extends THAT heading's title too — a 4-line
            # government letterhead is one heading, not four; (3) anywhere
            # else — e.g. mid-way through an unrelated numbered list, which
            # is what produced the cross-page merge bug this fixes — it
            # closes whatever is open and starts a fresh node.
            stripped_text = block.text.strip()
            if match is None and _is_allcaps_heading(stripped_text):
                is_heading_continuation = (
                    last_standalone_heading_id is not None
                    and last_standalone_heading_id in nodes
                    and last_block_bottom is not None
                    and (block.top - last_block_bottom) <= STANDALONE_HEADING_MERGE_GAP
                )
                if pending_heading_target_id is not None and pending_heading_target_id in nodes:
                    target = nodes[pending_heading_target_id]
                    target.title = f"{target.title} {stripped_text}".strip() if target.title else stripped_text
                    last_standalone_heading_id = None
                elif is_heading_continuation:
                    target = nodes[last_standalone_heading_id]
                    target.title = f"{target.title} {stripped_text}".strip() if target.title else stripped_text
                else:
                    node_id = id_gen.next()
                    node = Node(
                        node_id=node_id,
                        parent_id=None,
                        depth=0,
                        node_type="heading",
                        label=None,
                        label_normalized=None,
                        numbering_style=None,
                        # The heading's own text lives in `title`, matching
                        # every other heading-bearing node type — not
                        # text_raw, which is reserved for the BODY that
                        # follows (e.g. a specimen form's opening line).
                        # Merging both into text_raw made a heading node
                        # unsearchable by its own title text.
                        title=stripped_text,
                        path=[],
                        text_raw="",
                        pages=[page],
                        bbox={"page": page, "x0": block.x0, "top": block.top, "x1": block.x1, "bottom": block.bottom},
                        char_span={"page": page, "start": start, "end": end},
                        reading_order=order_gen.next(),
                        extraction={"method": "native", "confidence": 1.0, "detector": "standalone_allcaps_heading", "flags": []},
                    )
                    nodes[node_id] = node
                    root_order.append(node_id)
                    # Known v1 limitation: this resets the whole ancestor
                    # stack to just this heading, so if a document (unlike
                    # this one) had further nested numbered content directly
                    # after such a heading with nothing shallower in between,
                    # it would attach under the heading instead of its real
                    # parent. Not the case anywhere in this document — the
                    # pattern only occurs at section boundaries here.
                    stack = [node]
                    last_standalone_heading_id = node_id
                pending_heading_target_id = None
                last_block_bottom = block.bottom
                prev_block_had_terminal = False
                continue
            pending_heading_target_id = None
            last_standalone_heading_id = None
            last_block_bottom = block.bottom

            # continuation text: attach to deepest open node, or start an
            # implicit paragraph node if nothing is open yet (e.g. preamble).
            if stack:
                leaf = stack[-1]
                if leaf.text_raw and not leaf.text_raw.endswith(" "):
                    leaf.text_raw += " "
                leaf.text_raw += block.text
                if page not in leaf.pages:
                    leaf.pages.append(page)
                    leaf.spans_page_break = leaf.spans_page_break or (page != leaf.pages[0])
                leaf.char_span["end"] = end
                prev_block_had_terminal = block.text.rstrip().endswith(TERMINAL_PUNCT)
            else:
                node_id = id_gen.next()
                node = Node(
                    node_id=node_id,
                    parent_id=None,
                    depth=0,
                    node_type="preamble",
                    label=None,
                    label_normalized=None,
                    numbering_style=None,
                    title=None,
                    path=[],
                    text_raw=block.text,
                    pages=[page],
                    bbox={"page": page, "x0": block.x0, "top": block.top, "x1": block.x1, "bottom": block.bottom},
                    char_span={"page": page, "start": start, "end": end},
                    reading_order=order_gen.next(),
                    extraction={"method": "native", "confidence": 1.0, "detector": "orphan_preamble", "flags": []},
                )
                nodes[node_id] = node
                root_order.append(node_id)
                stack.append(node)
                prev_block_had_terminal = block.text.rstrip().endswith(TERMINAL_PUNCT)

        # page-break stitching: handled naturally above since `stack` persists
        # across the page loop and continuation blocks with no numbering merge
        # into the still-open leaf. We only need to guard against a *new*
        # heading being misread as continuation, which match_numbering already
        # prevents by firing before the fallback branch.
        _ = prev_block_had_terminal

    return list(nodes.values()), page_raw_text, quality_flags
