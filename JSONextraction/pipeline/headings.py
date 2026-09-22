"""Is this line a heading? Scored from several signals, not from its case.

`tree.py` used one rule: an unnumbered ALL-CAPS line of ten or more characters
is a title, and it reset the whole ancestor stack. Real contracts also set party
names, emphasis and whole paragraphs in capitals, so one such line in the middle
of a clause detached every unit after it — measured on the authored
`parts_toc_and_caps` specimen, an ALL-CAPS sentence inside a clause body
orphaned the ayat that followed it.

Capitals are now one signal among several. A heading is short, set apart from
what surrounds it, and often centred or bold; a capitalised *sentence* inside a
body is long, flush with the body, and continues the paragraph above it.

Bold is unavailable on the OCR path (`is_bold` is always False there), so its
weight is small and geometry carries the decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Signal weights. They sum to more than the threshold on purpose: a heading
# normally shows several of these, and no single one is sufficient.
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
# A division heading — one that may open a top-level unit — needs to look set
# apart, not merely be short and capitalised.
DIVISION_THRESHOLD = 0.75

MAX_HEADING_WORDS = 12
# A line of at least this many words ending in a full stop is a sentence.
MIN_SENTENCE_WORDS = 6
# A line of at least this many words ending in a colon introduces what follows
# ("Kami yang bertanda tangan di bawah ini:"). A one-word "CATATAN:" still reads
# as a heading.
MIN_LEAD_IN_WORDS = 4
# A division names a part of the document, so it is never a single token: a
# body fragment like "SSKK;" is not a new part of the contract.
MIN_DIVISION_WORDS = 2
ISOLATION_LINE_HEIGHTS = 1.2
CENTRE_TOLERANCE_FRACTION = 0.06
CENTRED_MAX_WIDTH_FRACTION = 0.7
LARGER_FONT_RATIO = 1.15

_TERMINAL_PUNCT = (".", "!", "?", ";")
# "Perihal : Penunjukan Penyedia untuk ...", "Nomor : 027/SP/2024" — a labelled
# field, which letterheads often set in bold. It states a value; a heading names
# a division.
_LABELLED_FIELD_RE = re.compile(r"^[^\W\d_][\w\s./-]{0,30}:\s*\S")
_SENTENCE_TAIL_RE = re.compile(r"[.!?;]\s*$")


@dataclass
class PageStats:
    """What the rest of the page looks like, so one line can be compared to it."""

    width: float
    line_height: float
    median_font_size: float
    body_x0: float           # the column most body lines start at

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
        """Some visual distinction from the body: capitals, bold, a larger face
        or centring. Being short and set apart is not enough — a letter is full
        of short isolated lines ("Perihal : ...", a signature block) that are
        not headings."""
        return any(s in self.signals for s in ("all_caps", "bold", "larger_font", "centred"))

    @property
    def is_heading(self) -> bool:
        return self.score >= HEADING_THRESHOLD and self.has_emphasis

    @property
    def is_division(self) -> bool:
        """Strong enough to close everything open above it. Deliberately harder
        than `is_heading`: getting this wrong detaches the rest of the document,
        which is what a stray "SSKK;" fragment did to 25 clauses."""
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
    """How strongly this line reads as a heading rather than body text."""
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
    # A heading does not continue the sentence above it.
    if previous_text is None or _SENTENCE_TAIL_RE.search(previous_text.strip() or "."):
        signals["starts_fresh"] = WEIGHTS["starts_fresh"]
    if stats.width:
        centre_offset = abs(((x0 + x1) / 2.0) - stats.width / 2.0)
        # Narrower than the column as well as mid-page: a full-width body line
        # is "centred" by accident, and that false signal alone was enough to
        # make wrapped prose read as a heading.
        is_narrow = (x1 - x0) <= CENTRED_MAX_WIDTH_FRACTION * stats.width
        if is_narrow and centre_offset <= CENTRE_TOLERANCE_FRACTION * stats.width and x0 > stats.body_x0 + 1.0:
            signals["centred"] = WEIGHTS["centred"]
    if font_size and stats.median_font_size and font_size >= LARGER_FONT_RATIO * stats.median_font_size:
        signals["larger_font"] = WEIGHTS["larger_font"]
    if is_bold:
        signals["bold"] = WEIGHTS["bold"]
    # A heading names a thing; a sentence states one and closes with a stop.
    # This is what separates a real heading from a capitalised sentence inside
    # a clause body, which scores on capitals alone.
    if not stripped.endswith((".", "!", "?")):
        signals["no_sentence_end"] = WEIGHTS["no_sentence_end"]

    # A line ending in a semicolon is one item of an enumeration, continuing
    # the sentence that introduced it — "... dituangkan dalam SSKK;".
    if stripped.endswith(";"):
        signals.clear()

    # A line ending mid-sentence is body text whatever else it looks like, and
    # a labelled field is a value, not a division.
    is_sentence = stripped.endswith((".", "!", "?")) and len(words) >= MIN_SENTENCE_WORDS
    is_lead_in = stripped.endswith(":") and len(words) >= MIN_LEAD_IN_WORDS
    if (
        stripped.endswith(",")
        or (len(words) > MAX_HEADING_WORDS and not stripped.endswith(_TERMINAL_PUNCT))
        or _LABELLED_FIELD_RE.match(stripped)
        # A run of words closed by a full stop is a sentence, however it is set.
        # Capitals alone made such a line a heading, and it then detached every
        # unit that followed.
        or is_sentence
        or is_lead_in
    ):
        signals.clear()

    return HeadingScore(round(sum(signals.values()), 4), signals, len(words))
