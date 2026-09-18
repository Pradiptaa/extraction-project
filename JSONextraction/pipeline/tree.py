"""Stage 6 — Numbering Detection & Tree Build. Sequence breaks are flagged,
not re-parsed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .blocks import TextBlock
from .numbering import match_numbering, sibling_successor
from .schema import NodeIdGenerator, ReadingOrderCounter

TERMINAL_PUNCT = (".", "!", "?", ":", ";", "…")

# An all-caps line with enough real letters is always a title here, never prose.
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
    sub_document: str | None = None


def _classify(style: str, label: str, column_index: int, layout_type: str, is_clause_scope_page: bool) -> tuple[str, int]:
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
        # "1." is a clause only inside the clause-bearing sub-document; elsewhere
        # it is a flat list item. Keyed on sub-document, never page geometry.
        return ("clause", 1) if is_clause_scope_page else ("list_item", 1)
    if style == "decimal_dotted":
        dots = label.count(".")
        return "subclause", 1 + dots
    if style == "paren_digit_both":
        # Also location-dependent: a nested condition list inside the clause
        # sub-document, an ayat directly under its Pasal elsewhere.
        return ("list_item", 3) if is_clause_scope_page else ("subclause", 1)
    if style in ("latin_lower", "roman_lower"):
        return "list_item", 3
    if style in ("paren_digit", "paren_latin"):
        return "list_item", 4
    if style == "bullet":
        return "list_item", 5
    return "paragraph", 2


def build_tree(
    pages_blocks: dict[int, list[TextBlock]],
    layout_by_page: dict[int, str],
    page_order: list[int],
    sub_document_by_page: dict[int, str | None] | None = None,
    clause_sub_document: str | None = None,
) -> tuple[list[Node], dict[int, str], list[str]]:
    """Returns (nodes, page_raw_text_by_page, quality_flags)."""
    sub_document_by_page = sub_document_by_page or {}
    id_gen = NodeIdGenerator()
    order_gen = ReadingOrderCounter()
    nodes: dict[str, Node] = {}
    stack: list[Node] = []  # open ancestors, shallow-to-deep
    quality_flags: list[str] = []
    page_raw_text: dict[int, str] = {}
    page_cursor: dict[int, int] = {}
    last_label_by_parent_style: dict[tuple[str | None, str], str] = {}
    root_order: list[str] = []
    # Most recent clause, so its wrapped left-column heading lines route to its
    # title rather than being spliced into the body mid-sentence.
    current_clause_id: str | None = None
    # Last part/section/article, whose title the next standalone ALL-CAPS line
    # extends. Cleared on every other block.
    pending_heading_target_id: str | None = None

    def cursor_append(page: int, text: str) -> tuple[int, int]:
        buf = page_raw_text.get(page, "")
        start = page_cursor.get(page, 0)
        page_raw_text[page] = buf + text + "\n"
        end = start + len(text)
        page_cursor[page] = end + 1
        return start, end

    # Vertical gap (points) within which consecutive blocks are one wrapped element.
    RULED_TABLE_CAPTION_GAP = 20.0
    STANDALONE_HEADING_MERGE_GAP = 20.0
    # A running header passes the gap check but must never merge with a neighbor.
    _RUNNING_HEADER_RE = re.compile(r"^LAMPIRAN\s+[A-Z]\b")

    for page in page_order:
        layout_type = layout_by_page.get(page, "single_column")
        blocks = pages_blocks.get(page, [])
        is_clause_scope_page = (
            clause_sub_document is not None
            and sub_document_by_page.get(page) == clause_sub_document
        )

        if layout_type == "ruled_table":
            # Blocks outside every table bbox are scattered captions/footnotes,
            # not prose: each becomes its own node rather than going through the
            # stack machinery, which would glue unrelated ones together.
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
                    node_type, depth = _classify(match.style, match.label_normalized, block.column_index, layout_type, is_clause_scope_page)
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
        # Reset per page: nothing on one page merges into a heading on another.
        last_standalone_heading_id: str | None = None
        last_block_bottom: float | None = None

        for block in blocks:
            start, end = cursor_append(page, block.text)
            match = match_numbering(block.text)

            if match is not None:
                node_type, depth = _classify(match.style, match.label_normalized, block.column_index, layout_type, is_clause_scope_page)
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
                # For a heading-bearing type the remainder is the title's first
                # line, not body — it goes to `title` only, never text_raw.
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
                pending_heading_target_id = node_id if node_type in ("part", "section", "article") else None
                last_standalone_heading_id = None
                last_block_bottom = block.bottom
                prev_block_had_terminal = node.text_raw.rstrip().endswith(TERMINAL_PUNCT) if node.text_raw else False
                continue

            # Checked before the ALL-CAPS rule below: an all-caps fragment in a
            # clause title must still route to the clause.
            if (
                layout_type == "two_column"
                and block.column_index == 0
                and current_clause_id is not None
                and current_clause_id in nodes
            ):
                # Unnumbered column-0 line under an open clause continues its
                # heading; body text lives in column 1.
                clause_node = nodes[current_clause_id]
                clause_node.title = f"{clause_node.title} {block.text}".strip() if clause_node.title else block.text
                pending_heading_target_id = None
                last_standalone_heading_id = None
                last_block_bottom = block.bottom
                prev_block_had_terminal = block.text.rstrip().endswith(TERMINAL_PUNCT)
                continue

            # An unnumbered ALL-CAPS line is always a title. It extends a pending
            # heading's title, else an adjacent standalone heading's, else starts
            # a fresh node.
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
                    # Known limitation: resets the whole ancestor stack, so
                    # nested content right after such a heading reparents to it.
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

        # Page-break stitching needs no work here: `stack` persists across pages.
        _ = prev_block_had_terminal

    return list(nodes.values()), page_raw_text, quality_flags
