"""Generic numbering-token recognizer, ordered by specificity."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ROMAN_RE = r"[ivxlcdm]+"


@dataclass(frozen=True)
class NumberingMatch:
    style: str
    label: str            # as printed, e.g. "37.2"
    label_normalized: str
    depth_hint: int        # coarse depth rank; refined later with indent/font
    remainder: str          # text after the numbering token


# Ordered most-specific first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Case-sensitive: lowercase "pasal 44.2" is an inline cross-reference, not a heading.
    ("chapter_word", re.compile(r"^\s*(BAB|Bab|BAGIAN|Bagian)\s+([IVXLCDM]+|\d+)\b[.:]?\s*")),
    # \d{1,3} bounds the match so Civil Code citations ("Pasal 1266") aren't read as headings.
    ("article_word", re.compile(r"^\s*(PASAL|Pasal)\s+(\d{1,3})\b[.:]?\s*")),
    ("letter_dotted", re.compile(r"^\s*([A-Z])\.(\d{1,2})\s+(?=[A-Z])")),
    # Two or more roman characters cannot be a section letter, so this is
    # unambiguous. A single "I."/"V."/"X." is not, and is resolved by its series
    # in `resolve_roman_series` — see there.
    ("roman_upper", re.compile(r"^\s*([IVXLCDM]{2,7})\.\s+(?=[A-Z0-9])")),
    ("letter_upper", re.compile(r"^\s*([A-Z])\.\s+(?=[A-Z])")),
    ("decimal_dotted", re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){1,4})\.?\s+")),
    ("decimal_plain", re.compile(r"^\s*(\d{1,3})\.\s+")),
    # Ayat numbering "(2)", distinct from paren_digit "2)" used for deep flat lists.
    ("paren_digit_both", re.compile(r"^\s*\((\d{1,3})\)\s+")),
    ("paren_digit", re.compile(r"^\s*(\d{1,3})\)\s+")),
    ("latin_lower", re.compile(r"^\s*([a-z])\.\s+")),
    ("paren_latin", re.compile(r"^\s*([a-z])\)\s+")),
    ("roman_lower", re.compile(rf"^\s*({ROMAN_RE})\.\s+", re.IGNORECASE)),
    ("bullet", re.compile(r"^\s*[•▪◦\-–]\s+")),
]

_BASE_DEPTH = {
    "chapter_word": 0,
    "article_word": 0,
    "letter_dotted": 1,
    "roman_upper": 0,
    "letter_upper": 1,
    "decimal_plain": 1,
    "paren_digit_both": 1,
    "paren_digit": 3,
    "latin_lower": 2,
    "paren_latin": 3,
    "roman_lower": 2,
    "bullet": 4,
}


def match_numbering(text: str) -> Optional[NumberingMatch]:
    if not text:
        return None
    for style, pattern in _PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        remainder = text[m.end():].strip()
        if style == "decimal_dotted":
            label = m.group(1)
            dot_count = label.count(".")
            depth = dot_count
            return NumberingMatch("decimal_dotted", label, label, depth, remainder)
        if style in ("chapter_word", "article_word"):
            label = m.group(0).strip().rstrip(".:")
            return NumberingMatch(style, label, label.upper(), _BASE_DEPTH[style], remainder)
        if style == "letter_dotted":
            label = f"{m.group(1)}.{m.group(2)}"
            return NumberingMatch(style, label, label.upper(), _BASE_DEPTH[style], remainder)
        if style == "bullet":
            return NumberingMatch(style, "•", "•", _BASE_DEPTH[style], remainder)
        label = m.group(1)
        return NumberingMatch(style, label, label.lower(), _BASE_DEPTH[style], remainder)
    return None


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def roman_to_int(label: str) -> Optional[int]:
    total, previous = 0, 0
    for char in reversed(label.upper()):
        value = _ROMAN_VALUES.get(char)
        if value is None:
            return None
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total or None


def int_to_roman(number: int) -> str:
    out = []
    for value, numeral in _ROMAN_NUMERALS:
        while number >= value:
            out.append(numeral)
            number -= value
    return "".join(out)


def resolve_roman_series(matches: list[Optional[NumberingMatch]]) -> list[Optional[NumberingMatch]]:
    """Re-reads single-character `letter_upper` markers that are really roman.

    "I." and "V." are both a section letter and a roman numeral, and the shape
    alone cannot tell them apart — which is why a document numbered I, II, III
    used to produce `letter_upper` for I and V but `roman_upper` for II, III and
    IV: one series split across two styles and two depths.

    Resolved by the series a marker sits in, over the whole document in reading
    order: a marker is roman when its neighbour in that series is the roman
    predecessor or successor ("I." before "II.", "V." after "IV."). A marker in
    a plain alphabetic run ("...G., H., I., J...") stays a letter.
    """
    resolved = list(matches)
    candidate_positions = [
        i for i, m in enumerate(resolved)
        if m is not None and m.style in ("letter_upper", "roman_upper")
    ]

    for index, position in enumerate(candidate_positions):
        match = resolved[position]
        if match.style != "letter_upper" or match.label.upper() not in _ROMAN_VALUES:
            continue
        value = roman_to_int(match.label)

        # An alphabetic run decides it: a marker directly after "H." is the
        # letter I, whatever else the document contains. Checked on the
        # immediate neighbours, since that is where a run shows itself.
        letter_evidence = False
        for offset, direction in ((-1, -1), (1, 1)):
            neighbour_index = index + offset
            if not 0 <= neighbour_index < len(candidate_positions):
                continue
            neighbour = resolved[candidate_positions[neighbour_index]]
            if len(neighbour.label) != 1 or not neighbour.label.isalpha():
                continue
            if neighbour.label.upper() == chr(ord(match.label.upper()) + direction) \
                    and neighbour.label.upper() not in _ROMAN_VALUES:
                letter_evidence = True

        # Otherwise the series decides, searched document-wide: "I." and its
        # "II." are separated by the whole of part I, so neighbouring markers
        # never see each other.
        roman_evidence = any(
            other.style == "roman_upper" and roman_to_int(other.label) in (value - 1, value + 1)
            for position_other in candidate_positions
            for other in (resolved[position_other],)
            if position_other != position
        )

        if roman_evidence and not letter_evidence:
            resolved[position] = NumberingMatch(
                "roman_upper", match.label, match.label.upper(),
                _BASE_DEPTH["roman_upper"], match.remainder,
            )
    return resolved


def sibling_successor(style: str, prev_label: str) -> Optional[str]:
    """Next expected sibling label, or None when the style has no successor."""
    if style in ("decimal_dotted", "decimal_plain", "article_word", "chapter_word", "paren_digit",
                 "paren_digit_both"):
        m = re.search(r"(\d+)$", prev_label)
        if m:
            n = int(m.group(1)) + 1
            prefix = prev_label[: m.start()]
            return f"{prefix}{n}"
        return None
    if style == "roman_upper":
        value = roman_to_int(prev_label)
        return int_to_roman(value + 1) if value else None
    if style in ("letter_upper",):
        if prev_label and prev_label[-1].isalpha():
            return prev_label[:-1] + chr(ord(prev_label[-1]) + 1)
        return None
    if style in ("latin_lower", "paren_latin"):
        if prev_label and prev_label[-1].isalpha():
            return prev_label[:-1] + chr(ord(prev_label[-1]) + 1)
        return None
    return None
