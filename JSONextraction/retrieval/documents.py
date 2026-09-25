"""Which contract a question names ("Pada file Rancangan Kontrak, ...").

`--document` says which specimen to search; this reads the same intent out of
the question itself, so the flag becomes optional rather than required. The
resolved scope is identical either way — only how it was arrived at differs.

Scoping to the wrong contract is the failure this whole feature exists to
prevent, so the rule is deliberately narrow: a document is only recognised when
the question *names* one, never when it merely resembles one. Two kinds of cue
do that, and they are treated differently on a miss:

  strong (`file`, `berkas`, an explicit `.pdf`)
      Nobody writes these by accident, so a name that matches nothing is
      reported back rather than ignored: the user asked about a file that is
      not in the corpus, and silently answering from all six would answer a
      different question.

  weak (`dokumen`, `kontrak`)
      Ordinary nouns in this corpus — "penyedia memutuskan kontrak secara
      sepihak" names no file. These scope when they match a real document and
      stay silent when they do not. Reporting their misses would break three of
      the twenty gate queries.

Ambiguity is refused under both, as in `store.resolve_scope`: two candidates
mean the question did not say which, and guessing is the thing to avoid.

Nothing here searches or ranks, so the retrievers and the gate are untouched.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from .registry import Matches

logger = logging.getLogger(__name__)

class Resolver(Protocol):
    """Built by `store.document_resolver`, which owns the choice between the
    indexed catalogue and a scan."""

    def __call__(self, text: str, limit: int = 5, filenames_only: bool = False) -> Matches: ...

# Cue word, then the candidate name up to a clause boundary. The leading
# preposition is optional so both "pada file X" and a bare "file X" match.
_PREFIX = r"(?:\b(?:pada|di|dalam|untuk|dari)\s+)?"
_TAIL = r"(?=[,?.:;]|$)"
_STRONG_RE = re.compile(rf"{_PREFIX}\b(?:file|berkas)\s+(?P<name>.{{2,60}}?){_TAIL}", re.IGNORECASE)
_WEAK_RE = re.compile(rf"{_PREFIX}\b(?:dokumen|kontrak)\s+(?P<name>.{{2,60}}?){_TAIL}", re.IGNORECASE)
# A bare filename, wherever it appears: "rehabGedung.pdf berapa ...".
_PDF_RE = re.compile(r"\b(?P<name>[\w][\w \-]{1,59}?)\.pdf\b", re.IGNORECASE)

@dataclass(frozen=True)
class DocumentMention:
    """What a question said about which contract to search.

    Exactly one of `scope`, `problem` is meaningful: a mention either resolved
    to one document or it did not. `remainder` is the question with the mention
    removed, and is only meaningful when `scope` is set.
    """

    scope: dict[str, str] = field(default_factory=dict)
    remainder: str = ""
    raw: str = ""
    # Set when the question named a document that cannot be searched: either no
    # such file, or more than one. `ask` prints this and stops.
    problem: str = ""

    @property
    def found(self) -> bool:
        return bool(self.scope)


def _named_span(span: str, resolve: Resolver, anchored: bool = True,
                filenames_only: bool = False) -> tuple[Matches, int, int]:
    """Find the words inside `span` that name a document.

    The name is not the whole span: a cue runs straight into the rest of the
    question with no punctuation between ("dokumen rehabGedung berapa lama
    ..."), so each leading run of words is tried and the best wins.

    A run naming exactly one document beats one naming several, and a longer
    run beats a shorter — "pembangunan" matches two specimens, "pembangunan
    rumah" matches one.

    `anchored` keeps the name against the front of the span, where a cue word
    puts it. Only a bare `.pdf` match needs it off, because that pattern can
    swallow the word before the filename ("file pembangunanSayap.pdf"). Letting
    a *cue* skip leading words would be far too loose: "daftar dokumen yang
    merupakan satu kesatuan bagian kontrak" would find `kontrak` several words
    in and refuse a perfectly ordinary question as ambiguous.

    Returns the matches and the run's character bounds within `span`, so the
    caller strips exactly those words and leaves the question intact.
    """
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
    """The candidates, as lines a reader can tell two namesakes apart by.

    Never the catalogue: a few lines carrying the contract number, the
    organisation and the year where those are known, then a count of what is
    not shown. At a thousand documents, printing them all explains nothing.
    """
    lines = [f"  {document.describe()}" for document in matches.documents]
    if matches.truncated:
        lines.append(f"  ...and {matches.truncated} more")
    return "\n".join(lines)


def parse(question: str, resolve: Resolver) -> DocumentMention:
    """Read the document a question names, if it names one.

    `resolve` comes from `store.document_resolver`, which decides whether the
    indexed catalogue or a scan answers. It is only ever called once a cue has
    fired, because most questions name no document and resolution should cost
    nothing when there is nothing to resolve.
    """
    # `is_filename` patterns carry their own extension, which must go with the
    # name; the cue patterns stop at the name itself.
    for pattern, strong, is_filename in ((_PDF_RE, True, True), (_STRONG_RE, True, False),
                                         (_WEAK_RE, False, False)):
        match = pattern.search(question)
        if not match:
            continue
        name = match.group("name").strip()
        # A weak cue is matched on filenames alone. `dokumen` and `kontrak` are
        # ordinary words here, so what follows them is usually an ordinary
        # sentence — and organisation names are made of ordinary words too:
        # "dalam kontrak kerja konstruksi ini" would otherwise find "Satuan
        # Kerja Dinas Tenaga Kerja" and scope a question about every contract
        # to one of them, silently. Only a strong cue, which the user offered
        # as a name, is allowed to match a contract's number, title or parties.
        matches, name_start, name_end = _named_span(
            name, resolve, anchored=not is_filename, filenames_only=not strong,
        )
        # The cue and the name go; whatever followed the name is still the
        # question. `match.start()` covers the cue, `cut` the name — extended
        # to the whole match for a filename, so its `.pdf` goes too.
        offset = question.index(name, match.start())
        cut = match.end() if is_filename else offset + name_end

        if matches.total == 1:
            document = matches.documents[0]
            key, filename = document.document_key, document.filename
            # Removed from the query: "pada file Rancangan Kontrak" is words
            # that appear in every contract, so leaving them in would pull the
            # whole corpus up the BM25 ranking on the strength of the scope.
            remainder = (question[: match.start()] + " " + question[cut:]).strip(" ,;:")
            remainder = re.sub(r"\s{2,}", " ", remainder).strip()
            logger.info("question names document %r", filename)
            return DocumentMention(scope={key: filename}, remainder=remainder,
                                   raw=question[offset + name_start : cut].strip())

        if matches.total > 1:
            # Refused under both cue kinds: the question named a document and
            # did not say which, which is exactly when guessing does harm.
            return DocumentMention(
                raw=match.group(0).strip(),
                problem=f"{name!r} matches {matches.total} documents:\n{_describe(matches)}\n"
                        "Name it more fully, or use --document.",
            )

        if strong:
            # Not followed by the catalogue. Listing every document was the
            # only available explanation while there were six of them; it is
            # not one at a thousand, and the question named a file, so what
            # helps is knowing that file is not here.
            return DocumentMention(
                raw=match.group(0).strip(),
                problem=f"no document matches {name!r} — `--list-documents` shows what is "
                        "in the collection.",
            )
        # A weak cue that matched nothing named no file at all; carry on.
        logger.debug("weak cue %r matched no document — searching the whole corpus", name)

    return DocumentMention()
