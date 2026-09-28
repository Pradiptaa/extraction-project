from __future__ import annotations

from dataclasses import dataclass

INDENT_TOLERANCE_PT = 4.0

TOP_LEVEL_STYLES = ("chapter_word", "article_word")

HEADING_STYLES = TOP_LEVEL_STYLES + ("letter_upper", "roman_upper", "letter_dotted", "chapter_word")


@dataclass
class Level:
    style: str
    depth: int
    x0: float
    label: str


class RelativeDepth:

    def __init__(self, indent_tolerance: float = INDENT_TOLERANCE_PT) -> None:
        self.levels: list[Level] = []
        self.indent_tolerance = indent_tolerance

    # helpers 
    def _find_style(self, style: str, x0: float) -> int | None:
        for i in range(len(self.levels) - 1, -1, -1):
            level = self.levels[i]
            if level.style != style:
                continue
            if abs(level.x0 - x0) <= self.indent_tolerance:
                return i
            if x0 < level.x0 - self.indent_tolerance:
                return i
        return None

    def _dotted_parent(self, label: str) -> int | None:
        if "." not in label:
            return None
        prefix = label.rsplit(".", 1)[0]
        for i in range(len(self.levels) - 1, -1, -1):
            if self.levels[i].label == prefix:
                return i
        return None

    def reset(self) -> None:
        self.levels.clear()

    # decision 
    def depth_for(self, style: str, label: str, x0: float) -> int:
        if style in TOP_LEVEL_STYLES:
            self.levels = [Level(style, 0, x0, label)]
            return 0

        parent_index = self._dotted_parent(label)
        if parent_index is not None:
            depth = self.levels[parent_index].depth + 1
            self.levels = self.levels[: parent_index + 1] + [Level(style, depth, x0, label)]
            return depth

        existing = self._find_style(style, x0)
        if existing is not None:
            depth = self.levels[existing].depth
            self.levels = self.levels[:existing] + [Level(style, depth, x0, label)]
            return depth

        while (
            self.levels
            and self.levels[-1].style not in HEADING_STYLES
            and x0 < self.levels[-1].x0 - self.indent_tolerance
        ):
            self.levels.pop()

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
