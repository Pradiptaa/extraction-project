"""Quick lookup: answers core-field questions from `<pdf-stem>_raw.json`, with no search and no model call."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Callable

from vocabulary import for_all_profiles

from .config import PROJECT_DIR

logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_DIR / "output" / "raw"

# Marks a question about what a clause says, even when it names a field.
_CLAUSE_INTENT = re.compile(
    r"\b(kewajiban|hak|jika|apabila|bila|bagaimana|mengapa|kenapa|syarat|ketentuan|prosedur|"
    r"tata\s+cara|tanggung\s+jawab|wajib|berubah|perubahan|adendum|akibat|asuransi|"
    r"pemutusan|sengketa|diatur|mengatur|pasal|ayat|alamat|korespondensi)\b"
)
MAX_WORDS = 12

_QUANTITY = r"\b(berapa|nilai|besar(nya)?|lama(nya)?|jumlah)\b"

# Specific number subtypes before the generic number rule.
_RULES: list[tuple[str, str | None, re.Pattern]] = [
    ("key_numbers", "contract_value", re.compile(r"\b(nilai|harga)\s+(kontrak|pekerjaan)\b")),
    ("key_numbers", "masa_pelaksanaan",
     re.compile(_QUANTITY + r".*\b(masa|waktu|jangka\s+waktu)\s+pelaksanaan\b")),
    ("key_numbers", "masa_pemeliharaan", re.compile(_QUANTITY + r".*\bmasa\s+pemeliharaan\b")),
    ("key_numbers", "denda_keterlambatan", re.compile(_QUANTITY + r".*\bdenda\s+keterlambatan\b")),
    ("key_numbers", "denda_cacat_mutu", re.compile(_QUANTITY + r".*\bdenda\s+cacat\s+mutu\b")),
    ("key_numbers", None, re.compile(r"\b(angka|nilai|nominal)\s+(penting|utama|nominal)\b|\bangka\s+nominal\b")),
    ("contract_number", None, re.compile(r"\b(nomor|nomer|no)\s+(kontrak|surat\s+perjanjian|perjanjian|spk)\b")),
    ("contract_name", None,
     re.compile(r"\bnama\s+(kontrak|paket(\s+pekerjaan)?|pekerjaan|proyek)\b|\bjudul\s+kontrak\b")),
    ("parties", None, re.compile(
        r"\bsiapa\b.*\b(penyedia|pihak|ppk|ppkom|pejabat|penandatangan|perusahaan|kontraktor)\b"
        r"|\bnama\s+(perusahaan|penyedia|kontraktor|ppk|ppkom|pejabat|pihak)\b"
        r"|\bpara\s+pihak\b"
    )),
    ("key_dates", None, re.compile(
        r"\b(tanggal|kapan)\b.*\b(penting|kontrak|perjanjian|penandatanganan|ditandatangani)\b"
    )),
]

# Display names come from the vocabulary (profiles/base_id.json plus the
# profile), so a family that adds an amount subtype does not also need an edit
# here. `vocabulary` imports neither pipeline nor retrieval, so the
# one-directional rule between them is untouched.
_VOCAB = for_all_profiles()
FIELD_LABELS = dict(_VOCAB.get("field_display_labels") or {})
SUBTYPE_LABELS = dict(_VOCAB.get("subtype_display_labels") or {})

# Template blanks are correct extraction, so they report as unfilled, not as a miss.
_PLACEHOLDER = re.compile(r"\.{4,}|\[diisi|…{2,}")
_LOW_CONFIDENCE = 0.7


@dataclass(frozen=True)
class Route:
    field: str
    subtype: str | None = None

    @property
    def label(self) -> str:
        return SUBTYPE_LABELS.get(self.subtype or "", FIELD_LABELS[self.field])


def _normalize(question: str) -> str:
    return " ".join(re.sub(r"[^\w\s/-]", " ", question.lower()).split())


def route(question: str) -> Route | None:
    """The core field a question asks for, or None to send it to search."""
    text = _normalize(question)
    if not text or len(text.split()) > MAX_WORDS or _CLAUSE_INTENT.search(text):
        return None
    for field_name, subtype, pattern in _RULES:
        if pattern.search(text):
            return Route(field_name, subtype)
    return None


@dataclass
class DocumentAnswer:
    document_key: str
    name: str
    raw_file: str
    lines: list[str] = field(default_factory=list)
    status: str = "found"  # found | unfilled | unresolved | missing_raw


def load_raw_documents(raw_dir: Path | None = None) -> dict[str, tuple[Path, dict]]:
    """`document_key` (the raw file's `source.sha256`) -> (path, parsed raw file)."""
    directory = RAW_DIR if raw_dir is None else raw_dir
    documents: dict[str, tuple[Path, dict]] = {}
    if not directory.is_dir():
        logger.warning("no raw extraction directory at %s — quick lookup unavailable", directory)
        return documents
    # Only `*_raw.json`: the directory also holds older files carrying pre-fix values.
    for path in sorted(directory.glob("*_raw.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read %s for quick lookup (%s)", path.name, exc)
            continue
        key = (document.get("source") or {}).get("sha256")
        if key:
            documents[key] = (path, document)
    return documents


def _read_one(path: Path, key: str) -> tuple[Path, dict] | None:
    """One raw file, if it is still the one the registry says it is.

    A `raw_path` is only as good as the last load: the file can be moved,
    deleted, or overwritten by a re-extraction of a different PDF under the
    same name. So the file is trusted only when its own `source.sha256` is the
    key asked for — a mismatch is not an answer about the wrong contract, it is
    a reason to look elsewhere.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s for quick lookup (%s)", path.name, exc)
        return None
    if (document.get("source") or {}).get("sha256") != key:
        logger.warning("%s no longer holds document %s — looking for it elsewhere",
                       path.name, key[:12])
        return None
    return path, document


def raw_document_provider(settings=None, raw_dir: Path | None = None) -> RawFetch:
    """Fetch one raw extraction by `document_key`, reading only that file.

    `load_raw_documents` parses every raw file on disk to answer a question
    about one of them — 18.7 ms a document, per question. With a registry the
    path is looked up and a single file read; a scoped question then costs one
    file however large the corpus grows.

    Anything the registry cannot vouch for — no registry, a key it does not
    hold, a path that has gone stale — falls back to the full scan, built once
    and only if needed. Correctness never depends on the registry being
    current, because every file is checked against its key when it is read.
    """
    from . import registry  # deferred: keeps quick lookup importable on its own

    cache: dict[str, tuple[Path, dict] | None] = {}
    scanned: list[dict[str, tuple[Path, dict]]] = []
    # Every registered path, read in one query the first time one is needed.
    # Opening the registry per key made an unscoped question slower than the
    # scan it replaced; one query of document_key -> path stays cheap at a
    # thousand documents and leaves one file read per document asked about.
    paths: list[dict[str, Path]] = []

    def from_scan(key: str) -> tuple[Path, dict] | None:
        if not scanned:
            scanned.append(load_raw_documents(raw_dir))
        return scanned[0].get(key)

    def registered_paths() -> dict[str, Path]:
        if not paths:
            found: dict[str, Path] = {}
            if settings is not None:
                # Guarded: a registry that fails mid-read costs the scan below,
                # never the question.
                found = registry.read(
                    settings.db_path,
                    lambda c: registry.raw_paths(c, settings.collection),
                    default={},
                )
            paths.append(found)
        return paths[0]

    def from_registry(key: str) -> tuple[Path, dict] | None:
        raw_path = registered_paths().get(key)
        return _read_one(raw_path, key) if raw_path else None

    def fetch(key: str) -> tuple[Path, dict] | None:
        if key not in cache:
            cache[key] = from_registry(key) or from_scan(key)
        return cache[key]

    return fetch


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and (not value.strip() or bool(_PLACEHOLDER.search(value))))


def _caveat(confidence, flags) -> str:
    flags = flags or []
    low = isinstance(confidence, (int, float)) and confidence < _LOW_CONFIDENCE
    return "  (Need Review)" if low or "ambiguous" in flags or "review_required" in flags else ""


def _one_line(text) -> str:
    return " ".join(str(text or "").split())


def _format_number(entry: dict) -> str:
    amount, unit = entry.get("amount"), entry.get("unit")
    if entry.get("type") == "reference_number":
        shown = str(entry.get("value"))
    elif entry.get("currency") == "IDR":
        shown = "Rp" + f"{amount:,.0f}".replace(",", ".")
    elif unit == "ratio":
        shown = f"{amount * 100:g}%"
    elif unit:
        shown = f"{amount:g} {unit.replace('_', ' ')}"
    else:
        shown = f"{amount:g}"
    return f"{shown}  (teks: \"{_one_line(entry.get('raw'))[:80]}\")"


def _entry_label(entry: dict) -> str:
    kind, subtype = entry.get("type"), entry.get("subtype")
    if kind == "contract_value":
        return SUBTYPE_LABELS["contract_value"]
    if kind == "reference_number":
        return f"Nomor referensi {subtype or ''}".strip()
    return SUBTYPE_LABELS.get(subtype or "", (subtype or kind or "angka").replace("_", " ").capitalize())


def _numbers(entries: list[dict], subtype: str | None) -> tuple[list[str], str]:
    lines, seen, placeholder = [], set(), False
    for entry in entries:
        if subtype is None:
            # Unclassified amounts have no known role, so they are not "important".
            if entry.get("subtype") == "unclassified":
                continue
        elif subtype not in (entry.get("type"), entry.get("subtype")):
            continue
        if entry.get("amount") is None and entry.get("value") is None:
            placeholder = placeholder or "template_placeholder" in (entry.get("flags") or [])
            continue
        dedupe = (entry.get("type"), entry.get("subtype"), entry.get("amount"), entry.get("unit"), entry.get("value"))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        lines.append(f"{_entry_label(entry)}: {_format_number(entry)}{_caveat(entry.get('confidence'), entry.get('flags'))}")
    if lines:
        return lines, "found"
    return [], "unfilled" if placeholder else "unresolved"


def _parties(parties: list[dict]) -> tuple[list[str], str]:
    lines, seen, any_filled, placeholder = [], set(), False, False
    for party in parties:
        organization = (party.get("organization") or {}).get("value")
        representative = party.get("representative") or {}
        name, position = representative.get("name"), representative.get("position")
        role = _one_line(party.get("role_label") or party.get("role") or "Pihak")
        blank = _is_blank(organization) and _is_blank(name)
        dedupe = (role, None if blank else organization, None if blank else name)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        if blank:
            placeholder = placeholder or "template_placeholder" in (party.get("flags") or [])
            lines.append(f"{role}: (tidak terisi di dokumen)")
            continue
        any_filled = True
        who = "(organisasi tidak terisi)" if _is_blank(organization) else _one_line(organization)
        if not _is_blank(name):
            who += f" — diwakili {_one_line(name)}" + (f", {_one_line(position)}" if position else "")
        lines.append(f"{role}: {who}")
    if any_filled:
        return lines, "found"
    return [], "unfilled" if placeholder else "unresolved"


def _dates(dates: list[dict]) -> tuple[list[str], str]:
    lines, seen = [], set()
    for entry in dates:
        # Year-only entries are mostly regulation citations ("Nomor 2 Tahun 2017").
        if entry.get("precision") != "day" or not entry.get("date") or entry["date"] in seen:
            continue
        seen.add(entry["date"])
        lines.append(f"{entry['date']}  (teks: \"{_one_line(entry.get('raw'))}\")")
    if not lines:
        return [], "unresolved"
    lines.append("(catatan: jenis tanggal belum diklasifikasikan — periksa teks aslinya)")
    return lines, "found"


def answer_document(document: dict, target: Route) -> tuple[list[str], str]:
    entry = (document.get("core") or {}).get(target.field) or {}
    value = entry.get("value")
    if target.field in ("contract_name", "contract_number"):
        if value is None:
            return [], "unresolved"
        if _is_blank(value):
            return [], "unfilled"
        return [f"{_one_line(value)}{_caveat(entry.get('confidence'), entry.get('flags'))}"], "found"
    if target.field == "parties":
        return _parties(value or [])
    if target.field == "key_dates":
        return _dates(value or [])
    return _numbers(value or [], target.subtype)


RawFetch = Callable[[str], "tuple[Path, dict] | None"]


def lookup(target: Route, scope: dict[str, str],
           raw_documents: Mapping[str, tuple[Path, dict]] | RawFetch) -> list[DocumentAnswer]:
    """One answer per document in `scope` (`document_key` -> display name).

    `raw_documents` is either every raw file, already parsed, or a function
    fetching one by key (`raw_document_provider`). The function is what makes a
    scoped question read one file instead of all of them; a dict is accepted
    because `dict.get` has exactly that signature.
    """
    fetch = raw_documents.get if isinstance(raw_documents, Mapping) else raw_documents
    answers = []
    for key, name in scope.items():
        found = fetch(key)
        if found is None:
            answers.append(DocumentAnswer(key, name or key[:12], "", status="missing_raw"))
            continue
        path, document = found
        lines, status = answer_document(document, target)
        display = name or (document.get("source") or {}).get("file") or key[:12]
        answers.append(DocumentAnswer(key, display, path.name, lines, status))
    return answers


def has_answer(answers: list[DocumentAnswer]) -> bool:
    return any(a.status in ("found", "unfilled") for a in answers)


_STATUS_TEXT = {
    "unfilled": "(tidak terisi di dokumen — bagian template yang dikosongkan)",
    "unresolved": "(tidak ditemukan di data inti — coba --route search)",
    "missing_raw": "(file ekstraksi mentah tidak ada di output/raw)",
}


def render(target: Route, answers: list[DocumentAnswer]) -> str:
    out = [f"{target.label}:", ""]
    for answer in answers:
        out.append(f"{answer.name}:")
        if answer.lines:
            out.extend(f"  - {line}" for line in answer.lines)
        else:
            out.append(f"  {_STATUS_TEXT[answer.status]}")
    sources = sorted({a.raw_file for a in answers if a.raw_file})
    if sources:
        out.extend(["", f"Sumber: core.{target.field} — {', '.join(sources)}"])
    return "\n".join(out)
