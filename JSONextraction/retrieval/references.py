from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from vocabulary import for_all_profiles

from .retrievers import Hit, scope_filter

logger = logging.getLogger(__name__)

LEVEL_WORDS = {
    "bab": "bab",
    "bagian": "bagian",
    "pasal": "pasal", "psl": "pasal", "ps": "pasal",
    "ayat": "ayat", "ay": "ayat",
    "angka": "angka", "butir": "angka", "poin": "angka", "point": "angka",
    "huruf": "huruf",
    "klausul": "pasal", "klausula": "pasal",
    "article": "pasal", "section": "bagian", "clause": "pasal", "paragraph": "ayat",
}
_LETTER_LEVELS = {"huruf"}
_ROMAN_LEVELS = {"bab", "bagian"}

PART_HINTS = dict(for_all_profiles().get("part_hints") or {})

_LEVEL_ALTERNATION = "|".join(sorted(LEVEL_WORDS, key=len, reverse=True))
_NUMBER = r"[\(\[]?\s*(?P<value>[0-9]+(?:\.[0-9]+)*|[ivxlcdm]{1,7}|[a-z])\s*[\)\]]?"
_SEGMENT_RE = re.compile(rf"\b(?P<word>{_LEVEL_ALTERNATION})\b\.?\s*{_NUMBER}", re.IGNORECASE)
_TRAILING_RE = re.compile(r"\s*[\(\[]\s*(?P<value>[0-9]+(?:\.[0-9]+)*|[a-z])\s*[\)\]]")
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
_ANNEX_RE = re.compile(r"\blampiran\s+(?P<letter>[a-z])\b", re.IGNORECASE)
_PART_ALTERNATION = "|".join(rf"\b{re.escape(term)}\b" for term in sorted(PART_HINTS, key=len, reverse=True))
_PART_RE = re.compile(_PART_ALTERNATION, re.IGNORECASE)
_PART_NUMBER_RE = re.compile(rf"(?P<part>{_PART_ALTERNATION})\s+(?:pasal\s+)?(?P<value>[0-9]+(?:\.[0-9]+)*)",
                             re.IGNORECASE)

HEADING_TEXT_CHARS = 90
HEADING_NODE_TYPES = {"article", "heading", "section", "part", "clause", "caption", "header"}


@dataclass(frozen=True)
class Citation:
    """`segments` are canonical label values, outermost first: ["5", "3"]."""

    segments: tuple[str, ...]
    levels: tuple[str, ...]
    part: str | None
    remainder: str
    raw: str

    def __str__(self) -> str:
        parts = [f"{level} {value}" for level, value in zip(self.levels, self.segments)]
        return " ".join(parts).capitalize()


def _clean_remainder(question: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for start, end in spans:
        out.append(question[last:start])
        last = end
    out.append(question[last:])
    text = re.sub(r"[\s,;:.?!]+$", "", re.sub(r"\s+", " ", "".join(out)).strip())
    text = re.sub(r"(?i)\b(sebagaimana|yang)?\s*(tercantum|dimaksud|diatur|disebut(kan)?)?"
                  r"\s*(menurut|berdasarkan|sesuai( dengan)?|dalam|pada|di|dari|isi|bunyi)?\s*$",
                  "", text.strip())
    return re.sub(r"[\s,;:.?!]+$", "", text).strip()


def parse(question: str) -> Citation | None:
    if not question:
        return None
    segments: list[str] = []
    levels: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _SEGMENT_RE.finditer(question):
        if match.start() < position:
            continue
        level = LEVEL_WORDS[match.group("word").lower().rstrip(".")]
        value = match.group("value").lower()
        if _ROMAN_RE.match(value) and len(value) > 1 and level not in _ROMAN_LEVELS:
            continue
        if value.isalpha() and len(value) == 1 and level not in _LETTER_LEVELS and not segments:
            continue
        segments.append(value)
        levels.append(level)
        start, position = match.start(), match.end()
        while (trailing := _TRAILING_RE.match(question, position)) is not None:
            segments.append(trailing.group("value").lower())
            levels.append("")
            position = trailing.end()
        spans.append((start, position))

    if not segments and (named := _PART_NUMBER_RE.search(question)) is not None:
        segments.append(named.group("value").lower())
        levels.append("pasal")
        spans.append((named.start(), named.end()))
    if not segments:
        return None
    if len(segments) == 1 and segments[0].isalpha() and len(segments[0]) == 1:
        return None

    part = None
    if (annex := _ANNEX_RE.search(question)) is not None:
        part = f"annex_{annex.group('letter').lower()}"
        spans.append(annex.span())
    elif (hint := _PART_RE.search(question)) is not None:
        part = PART_HINTS[re.sub(r"\s+", " ", hint.group(0).lower())]
        spans.append(hint.span())

    raw = question[spans[0][0]:position].strip()
    return Citation(tuple(segments), tuple(levels), part,
                    _clean_remainder(question, sorted(spans)), raw)


def _normalize_segment(segment: str) -> tuple[str, str]:
    text = segment.strip().strip("()[]").lower()
    match = re.match(rf"^({_LEVEL_ALTERNATION})\b\.?\s*(.*)$", text, re.IGNORECASE)
    if match and match.group(2):
        return LEVEL_WORDS[match.group(1).lower()], match.group(2).strip()
    return "", text


def normalize_path(path: str) -> tuple[list[str], list[str]]:
    levels, values = [], []
    for segment in (path or "").split("/"):
        if not segment.strip():
            continue
        level, value = _normalize_segment(segment)
        levels.append(level)
        values.append(value)
    return levels, values


def _variants(segments: tuple[str, ...]) -> list[tuple[str, ...]]:
    forms = [segments]
    if len(segments) > 1 and all(re.fullmatch(r"[0-9.]+", value) for value in segments):
        forms.append((".".join(segments),))
    split: list[str] = []
    for value in segments:
        split.extend(value.split(".") if "." in value else [value])
    if tuple(split) != segments:
        forms.append(tuple(split))
    return forms


def match_tier(citation: Citation, path: str) -> int | None:
    levels, values = normalize_path(path)
    if not values:
        return None
    for form in _variants(citation.segments):
        size = len(form)
        if size > len(values) or tuple(values[-size:]) != form:
            continue
        wanted = citation.levels[-size:] if size <= len(citation.levels) else citation.levels
        found = levels[-size:]
        pairs = [(w, f) for w, f in zip(wanted, found) if w and f]
        if pairs and all(w == f for w, f in pairs):
            return 0
        return 1
    return None


@dataclass
class ReferenceResult:
    hits: list[Hit] = field(default_factory=list)
    tier: int | None = None
    part_relaxed: bool = False
    expanded: int = 0

    @property
    def found(self) -> bool:
        return bool(self.hits)


def _is_heading(hit: Hit) -> bool:
    node_type = str((hit.metadata or {}).get("node_type") or "")
    return node_type in HEADING_NODE_TYPES and len(hit.text or "") < HEADING_TEXT_CHARS


def find(citation: Citation, collection, scope: set[str] | None = None, limit: int = 5) -> ReferenceResult:
    got = collection.get(where=scope_filter(scope), include=["metadatas", "documents"])
    rows = list(zip(got.get("ids") or [], got.get("metadatas") or [], got.get("documents") or []))

    by_tier: dict[int, list[Hit]] = {0: [], 1: []}
    for row_id, metadata, text in rows:
        metadata = dict(metadata or {})
        tier = match_tier(citation, str(metadata.get("hierarchy_path") or ""))
        if tier is not None:
            by_tier[tier].append(Hit(id=row_id, score=0.0, metadata=metadata, text=text or ""))

    result = ReferenceResult()
    for tier in (0, 1):
        if not by_tier[tier]:
            continue
        hits = by_tier[tier]
        if citation.part:
            in_part = [hit for hit in hits if (hit.metadata or {}).get("sub_document") == citation.part]
            result.part_relaxed = not in_part
            hits = in_part or hits
        result.tier = tier
        result.hits = hits[:limit]
        break

    if result.found and all(_is_heading(hit) for hit in result.hits):
        result.hits, result.expanded = _expand(result.hits, rows, limit)
    return result


def _expand(headings: list[Hit], rows, limit: int) -> tuple[list[Hit], int]:
    out = list(headings)
    for heading in headings:
        metadata = heading.metadata or {}
        prefix = f"{metadata.get('hierarchy_path') or ''}/"
        document = metadata.get("document_key")
        children = [
            Hit(id=row_id, score=0.0, metadata=dict(child or {}), text=text or "")
            for row_id, child, text in rows
            if child
            and child.get("document_key") == document
            and str(child.get("hierarchy_path") or "").startswith(prefix)
        ]
        out.extend(sorted(children, key=lambda hit: str((hit.metadata or {}).get("hierarchy_path"))))
        if len(out) >= limit:
            break
    return out[:limit], max(0, len(out[:limit]) - len(headings))


def merge(pinned: list[Hit], searched: list[Hit], k: int) -> list[Hit]:
    seen = {hit.id for hit in pinned}
    out = list(pinned[:k])
    for hit in searched:
        if len(out) >= k:
            break
        if hit.id not in seen:
            seen.add(hit.id)
            out.append(hit)
    return out
