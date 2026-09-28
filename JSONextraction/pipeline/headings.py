from __future__ import annotations

import re
from dataclasses import dataclass, field

WEIGHTS = {
    "all_caps": 0.30,
    "short": 0.20,
    "isolated_above": 0.20,
    "starts_fresh": 0.15,
    "centred": 0.15,
    "no_sentence_end": 0.15,
    "larger_font": 0.15,
    "bold": 0.10,
}
HEADING_THRESHOLD = 0.60
DIVISION_THRESHOLD = 0.75

MAX_HEADING_WORDS = 12
MIN_SENTENCE_WORDS = 6
MIN_LEAD_IN_WORDS = 4
MIN_DIVISION_WORDS = 2
ISOLATION_LINE_HEIGHTS = 1.2
CENTRE_TOLERANCE_FRACTION = 0.06
CENTRED_MAX_WIDTH_FRACTION = 0.7
LARGER_FONT_RATIO = 1.15

_TERMINAL_PUNCT = (".", "!", "?", ";")
_LABELLED_FIELD_RE = re.compile(r"^[^\W\d_][\w\s./-]{0,30}:\s*\S")
_SENTENCE_TAIL_RE = re.compile(r"[.!?;]\s*$")


@dataclass
class PageStats:

    width: float
    line_height: float
    median_font_size: float
    body_x0: float 

    @classmethod
    def from_blocks(cls, blocks, page_width: float, line_height: float) -> "PageStats":
        sizes = sorted(b.font_size for b in blocks if b.font_size) or [0.0]
        x0s = sorted(b.x0 for b in blocks) or [0.0]
        return cls(
            width=page_width,
            line_height=line_height or 12.0,
            median_font_size=sizes[len(sizes) // 2],
            body_x0=x0s[len(x0s) // 2],
        )


@dataclass
class HeadingScore:
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    words: int = 0

    @property
    def has_emphasis(self) -> bool:
        return any(s in self.signals for s in ("all_caps", "bold", "larger_font", "centred"))

    @property
    def is_heading(self) -> bool:
        return self.score >= HEADING_THRESHOLD and self.has_emphasis

    @property
    def is_division(self) -> bool:
        return self.score >= DIVISION_THRESHOLD and self.words >= MIN_DIVISION_WORDS

    def __str__(self) -> str:
        return f"{self.score:.2f} ({', '.join(sorted(self.signals))})"


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0


def score_heading(
    text: str,
    *,
    stats: PageStats,
    x0: float,
    x1: float,
    font_size: float = 0.0,
    is_bold: bool = False,
    gap_above: float | None = None,
    previous_text: str | None = None,
) -> HeadingScore:
    stripped = text.strip()
    signals: dict[str, float] = {}
    if not stripped:
        return HeadingScore(0.0, signals, 0)

    words = stripped.split()
    if _caps_ratio(stripped) >= 0.9 and sum(c.isalpha() for c in stripped) >= 4:
        signals["all_caps"] = WEIGHTS["all_caps"]
    if len(words) <= MAX_HEADING_WORDS:
        signals["short"] = WEIGHTS["short"]
    if gap_above is not None and gap_above >= ISOLATION_LINE_HEIGHTS * stats.line_height:
        signals["isolated_above"] = WEIGHTS["isolated_above"]
    if previous_text is None or _SENTENCE_TAIL_RE.search(previous_text.strip() or "."):
        signals["starts_fresh"] = WEIGHTS["starts_fresh"]
    if stats.width:
        centre_offset = abs(((x0 + x1) / 2.0) - stats.width / 2.0)
        is_narrow = (x1 - x0) <= CENTRED_MAX_WIDTH_FRACTION * stats.width
        if is_narrow and centre_offset <= CENTRE_TOLERANCE_FRACTION * stats.width and x0 > stats.body_x0 + 1.0:
            signals["centred"] = WEIGHTS["centred"]
    if font_size and stats.median_font_size and font_size >= LARGER_FONT_RATIO * stats.median_font_size:
        signals["larger_font"] = WEIGHTS["larger_font"]
    if is_bold:
        signals["bold"] = WEIGHTS["bold"]
    if not stripped.endswith((".", "!", "?")):
        signals["no_sentence_end"] = WEIGHTS["no_sentence_end"]

    if stripped.endswith(";"):
        signals.clear()

    is_sentence = stripped.endswith((".", "!", "?")) and len(words) >= MIN_SENTENCE_WORDS
    is_lead_in = stripped.endswith(":") and len(words) >= MIN_LEAD_IN_WORDS
    if (
        stripped.endswith(",")
        or (len(words) > MAX_HEADING_WORDS and not stripped.endswith(_TERMINAL_PUNCT))
        or _LABELLED_FIELD_RE.match(stripped)
        or is_sentence
        or is_lead_in
    ):
        signals.clear()

    return HeadingScore(round(sum(signals.values()), 4), signals, len(words))
