"""Relative depth: how deep a numbered unit sits, learned per document.

`numbering._BASE_DEPTH` assigns each style one fixed depth — `latin_lower` is
always 3, `paren_digit` always 4 — which encodes the Perpres-16 nesting order
(`1.` > `1.1` > `a.` > `1)`). A document that nests `A.` > `1.` > `a.`, or
`I.` > `A.` > `1.`, gets inverted parent/child links: measured on the authored
`inverted_nesting` specimen, every unit's text was found but only 6.7% of them
landed on the right path.

Here the order is observed instead. The first time a style appears under an open
unit it becomes a child level; when it appears again it is a sibling of that
level. Indentation is the tiebreaker, not the rule, because a wrapped or
out-dented line must not invent a level.

Dotted labels ("21.4") keep their own rule: the label states its own parentage,
so it attaches to the open ancestor whose label is its prefix.

The engine is per document, so one file's ordering never leaks into another's.
"""
from __future__ import annotations

from dataclasses import dataclass

# A style must out-dent by more than this to pop a level, and in-dent by more
# than this to open one. In points, but compared against values already scaled
# by `layout.median_line_height`, so it tracks the page's own type size.
INDENT_TOLERANCE_PT = 4.0

# Styles that always open a top-level unit: a "BAB"/"Pasal"/"Article" heading
# names the document's own divisions, so it can never be a child of a list item.
TOP_LEVEL_STYLES = ("chapter_word", "article_word")

# Styles printed as a heading on their own line. What follows one is its
# content, which normally starts back at the same margin — so the equal-indent
# rule below must not read it as a sibling. A Surat Perjanjian is exactly this
# shape: "Pasal 5" on its own line, then "(1)", "(2)" at the left margin.
HEADING_STYLES = TOP_LEVEL_STYLES + ("letter_upper", "roman_upper", "letter_dotted", "chapter_word")


@dataclass
class Level:
    style: str
    depth: int
    x0: float
    label: str


class RelativeDepth:
    """Tracks the open numbering levels and answers `depth_for()` per marker."""

    def __init__(self, indent_tolerance: float = INDENT_TOLERANCE_PT) -> None:
        self.levels: list[Level] = []
        self.indent_tolerance = indent_tolerance

    # -- helpers ---------------------------------------------------------
    def _find_style(self, style: str, x0: float) -> int | None:
        """Index of an open level of this style at a compatible indent."""
        for i in range(len(self.levels) - 1, -1, -1):
            level = self.levels[i]
            if level.style != style:
                continue
            if abs(level.x0 - x0) <= self.indent_tolerance:
                return i
            # Same style at a clearly different indent is a different level:
            # "a." nested under "a." happens in deeply numbered annexes.
            if x0 < level.x0 - self.indent_tolerance:
                return i
        return None

    def _dotted_parent(self, label: str) -> int | None:
        """For "21.4", the open level whose label is "21"."""
        if "." not in label:
            return None
        prefix = label.rsplit(".", 1)[0]
        for i in range(len(self.levels) - 1, -1, -1):
            if self.levels[i].label == prefix:
                return i
        return None

    def reset(self) -> None:
        self.levels.clear()

    # -- the decision ----------------------------------------------------
    def depth_for(self, style: str, label: str, x0: float) -> int:
        if style in TOP_LEVEL_STYLES:
            self.levels = [Level(style, 0, x0, label)]
            return 0

        # A label that names its own parent ("21.4" under "21") is placed by the
        # label, whatever the indent says.
        parent_index = self._dotted_parent(label)
        if parent_index is not None:
            depth = self.levels[parent_index].depth + 1
            self.levels = self.levels[: parent_index + 1] + [Level(style, depth, x0, label)]
            return depth

        existing = self._find_style(style, x0)
        if existing is not None:
            # A sibling: everything opened below this level is closed.
            depth = self.levels[existing].depth
            self.levels = self.levels[:existing] + [Level(style, depth, x0, label)]
            return depth

        # A style not currently open: out-dent first, then decide by indent.
        # Heading levels are never popped on indent alone — a heading is often
        # centred ("Pasal 5" at x0=316) while its own ayat start at the left
        # margin (x0=113), so its x0 says nothing about what nests under it. A
        # heading closes when a heading of its own style recurs, or when a
        # top-level style resets everything.
        while (
            self.levels
            and self.levels[-1].style not in HEADING_STYLES
            and x0 < self.levels[-1].x0 - self.indent_tolerance
        ):
            self.levels.pop()

        # Nesting is expressed by indentation, so a marker starting in the same
        # column as the open level is its *sibling*, not its child — even when
        # the style differs. Without this, a body line that merely begins with
        # something numbering-shaped ("38.2 dan 38.3 hanya berlaku ...") opens a
        # level, and everything after it is buried one step too deep.
        if (
            self.levels
            and self.levels[-1].style not in HEADING_STYLES
            and abs(x0 - self.levels[-1].x0) <= self.indent_tolerance
        ):
            depth = self.levels[-1].depth
            self.levels = self.levels[:-1] + [Level(style, depth, x0, label)]
            return depth

        depth = self.levels[-1].depth + 1 if self.levels else 0
        self.levels.append(Level(style, depth, x0, label))
        return depth
