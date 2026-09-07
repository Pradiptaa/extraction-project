"""Generic numbering-token recognizer, ordered by specificity.

No document-specific vocabulary lives here — a profile decides which node_type
a given style maps to; this module only recognizes the *shape* of a label.
"""
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


# Ordered most-specific first. Each entry: (style_name, compiled_regex, depth_hint_fn)
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Case-sensitive, Title-Case-or-ALL-CAPS only: real "Pasal"/"BAB" headings
    # are printed capitalized in this document family, while inline
    # cross-references ("...sesuai pasal 44.2...") use all-lowercase and must
    # NOT be mistaken for a new heading — that misfire duplicated article
    # nodes in testing (a fully case-insensitive match caught both).
    ("chapter_word", re.compile(r"^\s*(BAB|Bab|BAGIAN|Bagian)\s+([IVXLCDM]+|\d+)\b[.:]?\s*")),
    ("article_word", re.compile(r"^\s*(PASAL|Pasal)\s+(\d+)\b[.:]?\s*")),
    # "B.2 Pengendalian Waktu" — a subsection inside lettered Part B, distinct
    # from both letter_upper ("B. TITLE", space right after the dot) and
    # decimal_dotted (starts with a digit, not a letter). Without this style,
    # these subsection headings match nothing and get silently absorbed as
    # continuation text into whatever clause happened to be open.
    ("letter_dotted", re.compile(r"^\s*([A-Z])\.(\d{1,2})\s+(?=[A-Z])")),
    ("letter_upper", re.compile(r"^\s*([A-Z])\.\s+(?=[A-Z])")),
    ("decimal_dotted", re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){1,4})\.?\s+")),
    ("decimal_plain", re.compile(r"^\s*(\d{1,3})\.\s+")),
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
    "letter_upper": 1,
    "decimal_plain": 1,
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
            depth = dot_count  # "37" -> 0 extra, "37.2" -> depth 1, "37.2.1" -> depth 2
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


def sibling_successor(style: str, prev_label: str) -> Optional[str]:
    """Given the previous sibling's label, what should the next one be? Used for the
    sequence validator. Returns None when the style has no well-defined successor
    (e.g. bullet)."""
    if style in ("decimal_dotted", "decimal_plain", "article_word", "chapter_word", "paren_digit"):
        m = re.search(r"(\d+)$", prev_label)
        if m:
            n = int(m.group(1)) + 1
            prefix = prev_label[: m.start()]
            return f"{prefix}{n}"
        return None
    if style in ("letter_upper",):
        if prev_label and prev_label[-1].isalpha():
            return prev_label[:-1] + chr(ord(prev_label[-1]) + 1)
        return None
    if style in ("latin_lower", "paren_latin"):
        if prev_label and prev_label[-1].isalpha():
            return prev_label[:-1] + chr(ord(prev_label[-1]) + 1)
        return None
    return None
