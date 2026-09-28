from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from .registry import Matches

logger = logging.getLogger(__name__)

class Resolver(Protocol):

    def __call__(self, text: str, limit: int = 5, filenames_only: bool = False) -> Matches: ...

_PREFIX = r"(?:\b(?:pada|di|dalam|untuk|dari)\s+)?"
_TAIL = r"(?=[,?.:;]|$)"
_STRONG_RE = re.compile(rf"{_PREFIX}\b(?:file|berkas)\s+(?P<name>.{{2,60}}?){_TAIL}", re.IGNORECASE)
_WEAK_RE = re.compile(rf"{_PREFIX}\b(?:dokumen|kontrak)\s+(?P<name>.{{2,60}}?){_TAIL}", re.IGNORECASE)
_PDF_RE = re.compile(r"\b(?P<name>[\w][\w \-]{1,59}?)\.pdf\b", re.IGNORECASE)

@dataclass(frozen=True)
class DocumentMention:

    scope: dict[str, str] = field(default_factory=dict)
    remainder: str = ""
    raw: str = ""
    problem: str = ""

    @property
    def found(self) -> bool:
        return bool(self.scope)


def _named_span(span: str, resolve: Resolver, anchored: bool = True,
                filenames_only: bool = False) -> tuple[Matches, int, int]:
    words = span.split()
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = span.index(word, cursor)
        bounds.append((start, start + len(word)))
        cursor = start + len(word)

    best: tuple[Matches, int, int] = (Matches(), 0, 0)
    best_rank = ()
    for first in (0,) if anchored else range(len(words)):
        for last in range(first, len(words)):
            matches = resolve(" ".join(words[first : last + 1]), filenames_only=filenames_only)
            if not matches.total:
                continue
            rank = (matches.total == 1, last - first + 1)
            if not best_rank or rank > best_rank:
                best_rank = rank
                best = (matches, bounds[first][0], bounds[last][1])
    return best


def _describe(matches: Matches) -> str:
    lines = [f"  {document.describe()}" for document in matches.documents]
    if matches.truncated:
        lines.append(f"  ...and {matches.truncated} more")
    return "\n".join(lines)


def parse(question: str, resolve: Resolver) -> DocumentMention:
    for pattern, strong, is_filename in ((_PDF_RE, True, True), (_STRONG_RE, True, False),
                                         (_WEAK_RE, False, False)):
        match = pattern.search(question)
        if not match:
            continue
        name = match.group("name").strip()
        matches, name_start, name_end = _named_span(
            name, resolve, anchored=not is_filename, filenames_only=not strong,
        )
        offset = question.index(name, match.start())
        cut = match.end() if is_filename else offset + name_end

        if matches.total == 1:
            document = matches.documents[0]
            key, filename = document.document_key, document.filename
            remainder = (question[: match.start()] + " " + question[cut:]).strip(" ,;:")
            remainder = re.sub(r"\s{2,}", " ", remainder).strip()
            logger.info("question names document %r", filename)
            return DocumentMention(scope={key: filename}, remainder=remainder,
                                   raw=question[offset + name_start : cut].strip())

        if matches.total > 1:
            return DocumentMention(
                raw=match.group(0).strip(),
                problem=f"{name!r} matches {matches.total} documents:\n{_describe(matches)}\n"
                        "Name it more fully, or use --document.",
            )

        if strong:
            return DocumentMention(
                raw=match.group(0).strip(),
                problem=f"no document matches {name!r} — `--list-documents` shows what is "
                        "in the collection.",
            )
        logger.debug("weak cue %r matched no document — searching the whole corpus", name)

    return DocumentMention()
