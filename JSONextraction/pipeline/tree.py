"""Stage 6 — Numbering Detection & Tree Build. Sequence breaks are flagged,
not re-parsed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .blocks import TextBlock
from .depth import RelativeDepth
from .headings import PageStats, score_heading
from .layout import REFERENCE_LINE_HEIGHT, median_line_height
from .numbering import match_numbering, resolve_roman_series, sibling_successor
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


def node_type_for(style: str, is_clause_scope_page: bool) -> str:
    """Stage B — what to *call* a unit. Depth is decided separately.

    Splitting the two is the point of the relative engine: the profile and the
    sub-document may change a unit's name without moving it in the tree.
    """
    if style == "chapter_word":
        return "part"
    if style == "article_word":
        return "article"
    if style in ("letter_upper", "roman_upper", "letter_dotted"):
        return "section"
    if style == "decimal_plain":
        # "1." is a clause only inside the clause-bearing sub-document; elsewhere
        # it is a flat list item. Keyed on sub-document, never page geometry.
        return "clause" if is_clause_scope_page else "list_item"
    if style == "decimal_dotted":
        return "subclause"
    if style == "paren_digit_both":
        return "list_item" if is_clause_scope_page else "subclause"
    if style in ("latin_lower", "roman_lower", "paren_digit", "paren_latin", "bullet"):
        return "list_item"
    return "paragraph"


def _classify(style: str, label: str, column_index: int, layout_type: str, is_clause_scope_page: bool) -> tuple[str, int]:
    """Legacy engine: one fixed depth per style. Returns (node_type, depth)."""
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
    if style in ("latin_lower", "roman_lower", "roman_upper"):
        # `roman_upper` is new in the relative engine; legacy keeps reading an
        # uppercase roman marker exactly as it read it before — as a depth-3
        # list item — so the switch changes nothing on this path.
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
    running_header_patterns: list[str] | None = None,
    engine: str = "legacy",
    page_width_by_page: dict[int, float] | None = None,
    sub_document_by_block: dict[tuple[int, int], str | None] | None = None,
) -> tuple[list[Node], dict[int, str], list[str]]:
    """Returns (nodes, page_raw_text_by_page, quality_flags).

    `engine` selects how deep a numbered unit sits: "legacy" is one fixed depth
    per numbering style, "relative" learns the document's own nesting order
    (see `depth.py`). Both name units the same way — only placement differs.
    """
    if engine not in ("legacy", "relative"):
        raise ValueError(f"unknown tree engine {engine!r} (expected legacy or relative)")
    relative_depth = RelativeDepth() if engine == "relative" else None
    page_width_by_page = page_width_by_page or {}
    # Block-level segmentation (relative engine): a part may start mid-page, so
    # a node takes the sub-document of the block it starts on.
    sub_document_by_block = sub_document_by_block or {}
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
    # The part the previous block belonged to, so a boundary can be seen.
    current_sub_document: str | None = None

    def cursor_append(page: int, text: str) -> tuple[int, int]:
        buf = page_raw_text.get(page, "")
        start = page_cursor.get(page, 0)
        page_raw_text[page] = buf + text + "\n"
        end = start + len(text)
        page_cursor[page] = end + 1
        return start, end

    # Vertical gap within which consecutive blocks are one wrapped element,
    # measured in line heights: 20.0 pt on this corpus's 12 pt text.
    CAPTION_GAP_RATIO = 20.0 / REFERENCE_LINE_HEIGHT
    HEADING_MERGE_GAP_RATIO = 20.0 / REFERENCE_LINE_HEIGHT
    # A running header passes the gap check but must never merge with a neighbor.
    # The patterns come from the profile ("^LAMPIRAN [A-Z]" for Perpres-16); with
    # no profile there are none, and such a line is an ordinary caption.
    header_patterns = [re.compile(p) for p in (running_header_patterns or [])]

    # Numbering is matched up front, in reading order, because "I." and "V." can
    # only be told from roman numerals by the series they sit in — which needs
    # the markers that come after them. The builder below then reads the
    # resolved match rather than re-matching.
    block_keys: list[tuple[int, int]] = []
    ordered_matches = []
    for page in page_order:
        for index, block in enumerate(pages_blocks.get(page, [])):
            block_keys.append((page, index))
            ordered_matches.append(match_numbering(block.text))
    if engine == "relative":
        # Resolution moves nodes ("I." joining its II./III. series instead of
        # reading as a section letter), so it belongs to the new engine. Legacy
        # stays bit-identical.
        ordered_matches = resolve_roman_series(ordered_matches)
    matches_by_key = dict(zip(block_keys, ordered_matches))

    for page in page_order:
        layout_type = layout_by_page.get(page, "single_column")
        blocks = pages_blocks.get(page, [])
        line_height = median_line_height([b.bottom - b.top for b in blocks])
        ruled_table_caption_gap = CAPTION_GAP_RATIO * line_height
        standalone_heading_merge_gap = HEADING_MERGE_GAP_RATIO * line_height
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
            for block_index, block in enumerate(blocks):
                start, end = cursor_append(page, block.text)
                match = matches_by_key[(page, block_index)]
                block_sub_document = sub_document_by_block.get((page, block_index), sub_document_by_page.get(page))
                is_running_header = match is None and any(p.match(block.text) for p in header_patterns)
                is_wrap = (
                    match is None
                    and not is_running_header
                    and last_node_id is not None
                    and last_node_id in nodes
                    and nodes[last_node_id].node_type != "header"
                    and last_bottom is not None
                    and (block.top - last_bottom) <= ruled_table_caption_gap
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
                    sub_document=block_sub_document,
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
        page_stats = PageStats.from_blocks(blocks, page_width_by_page.get(page, 595.0), line_height)

        for block_index, block in enumerate(blocks):
            start, end = cursor_append(page, block.text)
            match = matches_by_key[(page, block_index)]
            # From the block order itself, so every branch sees the real
            # predecessor; tracking it per branch left it stale and inflated
            # every heading score by the "starts fresh" signal.
            previous_block = blocks[block_index - 1] if block_index else None
            previous_block_text = previous_block.text if previous_block else None
            gap_above = block.top - previous_block.bottom if previous_block else None
            block_sub_document = sub_document_by_block.get((page, block_index), sub_document_by_page.get(page))
            block_in_clause_scope = (
                block_sub_document == clause_sub_document if sub_document_by_block
                else is_clause_scope_page
            )

            # A new sub-document is the strongest division there is: the SSUK
            # does not continue the Surat Perjanjian's numbering, it replaces
            # it. Without this the "II. SYARAT-SYARAT UMUM KONTRAK" heading
            # nested under whatever was open — measured as the whole SSUK
            # hanging off `Pasal 5/3` on two specimens.
            if (
                relative_depth is not None
                and sub_document_by_block
                and block_sub_document is not None
                and block_sub_document != current_sub_document
            ):
                relative_depth.reset()
                stack = []
                current_clause_id = None
                pending_heading_target_id = None
            current_sub_document = block_sub_document

            if match is not None:
                if relative_depth is not None:
                    # Stage A decides placement from the document's own order,
                    # Stage B decides the name. A right-column "1." in a
                    # gutter-label layout is a body sub-reference either way.
                    node_type = node_type_for(match.style, block_in_clause_scope)
                    if layout_type == "two_column" and block.column_index == 1 and match.style == "decimal_plain":
                        node_type = "subclause"
                    depth = relative_depth.depth_for(match.style, match.label_normalized, block.x0)
                else:
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
                    sub_document=block_sub_document,
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

            # An unnumbered line that reads as a title. It extends a pending
            # heading's title, else an adjacent standalone heading's, else starts
            # a fresh node.
            stripped_text = block.text.strip()
            if relative_depth is not None:
                # Scored from several signals (headings.py). A capitalised
                # *sentence* inside a clause body scores low and stays body text,
                # where the old case-only rule made it a heading and detached
                # everything after it.
                heading_score = score_heading(
                    stripped_text,
                    stats=page_stats,
                    x0=block.x0, x1=block.x1,
                    font_size=block.font_size, is_bold=block.is_bold,
                    gap_above=gap_above,
                    previous_text=previous_block_text,
                )
                looks_like_heading = heading_score.is_heading
            else:
                heading_score = None
                looks_like_heading = _is_allcaps_heading(stripped_text)
            if match is None and looks_like_heading:
                is_heading_continuation = (
                    last_standalone_heading_id is not None
                    and last_standalone_heading_id in nodes
                    and last_block_bottom is not None
                    and (block.top - last_block_bottom) <= standalone_heading_merge_gap
                )
                if pending_heading_target_id is not None and pending_heading_target_id in nodes:
                    target = nodes[pending_heading_target_id]
                    target.title = f"{target.title} {stripped_text}".strip() if target.title else stripped_text
                    last_standalone_heading_id = None
                elif is_heading_continuation:
                    target = nodes[last_standalone_heading_id]
                    target.title = f"{target.title} {stripped_text}".strip() if target.title else stripped_text
                else:
                    # A division heading stands at the top level and closes what
                    # came before. A weaker one — an emphasised line inside a
                    # clause — nests under the unit it appears in, so it becomes
                    # the attachment point for the text beneath it without
                    # detaching the units above it.
                    nests_inside = (
                        relative_depth is not None
                        and heading_score is not None
                        and not heading_score.is_division
                        and bool(stack)
                    )
                    heading_parent = stack[-1] if nests_inside else None
                    node_id = id_gen.next()
                    node = Node(
                        node_id=node_id,
                        parent_id=heading_parent.node_id if heading_parent else None,
                        depth=heading_parent.depth + 1 if heading_parent else 0,
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
                        extraction={"method": "native", "confidence": 1.0, "detector": "standalone_heading", "flags": []},
                        sub_document=block_sub_document,
                    )
                    nodes[node_id] = node
                    if heading_parent is not None:
                        heading_parent.children.append(node_id)
                    else:
                        root_order.append(node_id)
                    if nests_inside:
                        stack.append(node)
                    elif relative_depth is None or heading_score is None or heading_score.is_division:
                        # A division heading ("SYARAT-SYARAT UMUM KONTRAK")
                        # closes what came before. A weaker one — an emphasised
                        # line inside a clause — is recorded without detaching
                        # the units that follow it, which the old rule did for
                        # every capitalised line.
                        stack = [node]
                        if relative_depth is not None:
                            relative_depth.reset()
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
                    sub_document=block_sub_document,
                )
                nodes[node_id] = node
                root_order.append(node_id)
                stack.append(node)
                prev_block_had_terminal = block.text.rstrip().endswith(TERMINAL_PUNCT)

        # Page-break stitching needs no work here: `stack` persists across pages.
        _ = prev_block_had_terminal

    return list(nodes.values()), page_raw_text, quality_flags
