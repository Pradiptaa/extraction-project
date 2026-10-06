from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from .registry import Matches, normalize_name

logger = logging.getLogger(__name__)

class Resolver(Protocol):

    def __call__(self, text: str, limit: int = 5, filenames_only: bool = False) -> Matches: ...

_PREFIX = r"(?:\b(?:pada|di|dalam|untuk|dari)\s+)?"
_TAIL = r"(?=[,?;)]|[.:](?:\s|$)|$)"
_SEP = r"(?:\s*:\s*|\s+)"
_STRONG_RE = re.compile(rf"{_PREFIX}\b(?:file|berkas){_SEP}(?P<name>.{{2,60}}?){_TAIL}", re.IGNORECASE)
_WEAK_RE = re.compile(rf"{_PREFIX}\b(?P<cue>dokumen|kontrak){_SEP}(?P<name>.{{2,60}}?){_TAIL}", re.IGNORECASE)
_PARTY_RE = re.compile(
    rf"(?:\b(?:berdasarkan|menurut|pada|di|dalam|dari)\s+)?(?:\b(?:dokumen|kontrak|perjanjian)\s+)?(?:\byang\s+)?"
    rf"\b(?:ditandatangani|ditanda\s+tangani|disetujui|dibuat)\s+oleh{_SEP}(?P<name>.{{2,60}}?){_TAIL}",
    re.IGNORECASE,
)
_PDF_RE = re.compile(r"\b(?P<name>[\w][\w \-]{1,59}?)\.pdf\b", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?:\b(?:nomor|nomer|no\.?)(?:\s+(?:kontrak|spk|surat\s+perjanjian|perjanjian))?\s*:?\s*"
    r"|\b(?:kontrak|spk|perjanjian)\s+(?:nomor|nomer|no\.?)\s*:?\s*)"
    r"(?P<name>(?=[\w./-]*\d)(?=[\w./-]*[/.-])[\w/-]+(?:\.[\w/-]+)*)"
    r"|(?<![\w./-])(?P<bare>(?=[\w./-]*\d)(?=[\w./-]*/)[\w/-]+(?:\.[\w/-]+)*)",
    re.IGNORECASE,
)

_LEAD_IN_RE = re.compile(r"(?:\b(?:pada|di|dalam|untuk|dari|dengan|file|berkas|dokumen)\s+)+$", re.IGNORECASE)

_FIELD_BEFORE_RE = re.compile(r"\b(?:nomor|nomer|no|nilai|harga|nama|judul|tanggal)\.?\s+$", re.IGNORECASE)

_MIN_NAME = 3

_DESCRIPTOR_WORDS = frozenset(
    "pejabat penandatangan pembuat komitmen ppk ppkom kepala direktur direktris pimpinan wakil sah "
    "penyedia kontraktor konsultan pihak perusahaan badan usaha dinas pemerintah kementerian satuan kerja "
    "kantor bagian bidang balai instansi lembaga kabupaten kota provinsi yang dan atau dari".split()
)

@dataclass(frozen=True)
class DocumentMention:

    scope: dict[str, str] = field(default_factory=dict)
    remainder: str = ""
    raw: str = ""
    problem: str = ""
    note: str = ""

    @property
    def found(self) -> bool:
        return bool(self.scope)


def _named_span(span: str, resolve: Resolver, anchored: bool = True,
                filenames_only: bool = False, skip_descriptors: bool = False) -> tuple[Matches, int, int]:
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
            text = " ".join(words[first : last + 1])
            if len(normalize_name(text)) < _MIN_NAME:
                continue
            if skip_descriptors and all(w.lower().strip(".,") in _DESCRIPTOR_WORDS for w in words[first : last + 1]):
                continue
            matches = resolve(text, filenames_only=filenames_only)
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
        lines.append(f"  ...dan {matches.truncated} lainnya")
    return "\n".join(lines)


def _tidy(text: str) -> str:
    text = re.sub(r"\(\s*\)|\"\s*\"|'\s*'", " ", text)
    text = re.sub(r"\s+([?,.;:!])", r"\1", text)
    text = re.sub(r"[,;:]+(?=[?.!])", "", text)
    text = re.sub(r"\s+\b(?:dan|atau|serta)\b(?=[?.!]?\s*$)", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;:")


def _cues(pattern: re.Pattern, question: str):
    position = 0
    while match := pattern.search(question, position):
        yield match
        position = match.start() + 1


def _first_mention(question: str, resolve: Resolver) -> DocumentMention:
    problem: DocumentMention | None = None
    for pattern, strong, kind in ((_PDF_RE, True, "filename"), (_NUMBER_RE, False, "number"),
                                  (_PARTY_RE, True, "party"), (_STRONG_RE, True, "name"),
                                  (_WEAK_RE, False, "name")):
        for match in _cues(pattern, question):
            mention = _read_cue(question, match, resolve, strong, kind)
            if mention is None:
                continue
            if mention.found:
                return mention
            problem = problem or mention
    return problem or DocumentMention()


def parse(question: str, resolve: Resolver) -> DocumentMention:
    first = _first_mention(question, resolve)
    if not first.found:
        return first
    scope, remainder, raw = dict(first.scope), first.remainder, [first.raw]
    notes = [first.note] if first.note else []
    while True:
        more = _first_mention(remainder, resolve)
        if not more.found or set(more.scope) <= set(scope):
            break
        scope.update(more.scope)
        remainder = more.remainder
        raw.append(more.raw)
        if more.note:
            notes.append(more.note)
    if not re.search(r"\w", remainder):
        remainder = question
    return DocumentMention(scope=scope, remainder=remainder, raw=", ".join(raw), note="\n".join(notes))


_QUESTION_WORDS = frozenset(
    "apa apakah berapa siapa kapan bagaimana mengapa kenapa dimana mana sebutkan jelaskan tolong".split()
)


def _dropped_name_part(name: str, name_end: int) -> str:
    tail = name[name_end:].split()
    if not tail:
        return ""
    word = tail[0].strip(".,;:")
    if any(c.isdigit() for c in word) or (word[:1].isupper() and word.lower() not in _QUESTION_WORDS):
        return word
    return ""


def _read_number(question: str, match: re.Match, name: str,
                 resolve: Resolver) -> DocumentMention | None:
    needle = normalize_name(name)
    if len(needle) < _MIN_NAME:
        return None
    owners = [d for d in resolve(name, limit=50).documents
              if needle in normalize_name(d.contract_number or "")]
    if len(owners) != 1:
        return None
    document = owners[0]
    before = _LEAD_IN_RE.sub("", question[: match.start()])
    remainder = _tidy(before + " " + question[match.end():])
    logger.info("question names document %r by its contract number", document.filename)
    return DocumentMention(scope={document.document_key: document.filename},
                           remainder=remainder, raw=match.group(0).strip())


def _looks_like_a_name(text: str) -> bool:
    words = [w for w in re.split(r"[\s,]+", text) if w]
    return 0 < len(words) <= 5 and not any(w.lower().strip(".") in _DESCRIPTOR_WORDS for w in words)


def _read_cue(question: str, match: re.Match, resolve: Resolver, strong: bool,
              kind: str) -> DocumentMention | None:
    if kind == "number":
        return _read_number(question, match, match.group("name") or match.group("bare"), resolve)
    name = match.group("name").strip()
    matches, name_start, name_end = _named_span(
        name, resolve, anchored=kind not in ("filename", "party"),
        filenames_only=kind == "filename" or (kind == "name" and not strong),
        skip_descriptors=kind == "party",
    )
    if kind == "party" and matches.total != 1 and not _looks_like_a_name(name):
        logger.debug("%r after the signing cue is a description, not a name — not scoping", name)
        return None
    offset = question.index(name, match.start())

    if matches.total == 1:
        document = matches.documents[0]
        key, filename = document.document_key, document.filename
        if kind == "filename":
            before = _LEAD_IN_RE.sub("", question[: offset + name_start])
            cut = match.end()
        elif "cue" in match.re.groupindex and _FIELD_BEFORE_RE.search(question[: match.start("cue")]):
            before, cut = question[: match.start("name")], offset + name_end
        else:
            before, cut = question[: match.start()], offset + name_end
        remainder = _tidy(before + " " + question[cut:])
        logger.info("question names document %r", filename)
        note = ""
        dropped = _dropped_name_part(name, name_end) if kind == "name" else ""
        if dropped:
            typed = name[name_start:name_end] + " " + dropped
            note = f"Tidak ada dokumen bernama \"{typed}\" — menggunakan {filename}."
        return DocumentMention(scope={key: filename}, remainder=remainder,
                               raw=question[offset + name_start : cut].strip(), note=note)

    if matches.total > 1 and strong:
        return DocumentMention(
            raw=match.group(0).strip(),
            problem=f"\"{name}\" cocok dengan {matches.total} dokumen:\n{_describe(matches)}\n"
                    "Sebutkan nama dokumennya dengan lebih lengkap.",
        )

    if strong:
        return DocumentMention(
            raw=match.group(0).strip(),
            problem=f"Tidak ada dokumen bernama \"{name}\".",
        )
    logger.debug("weak cue %r named no single document — searching the whole corpus", name)
    return None
