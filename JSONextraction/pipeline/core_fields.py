"""Stage 8 — CORE FIELD RESOLUTION. Promotes entities into the six fields RAG
always needs, via the strategy cascade from
skema_json_dan_logika_ekstraksi.md 4.4-4.5.

v1 scope: strategies 1-4 (labeled lookup, contextual pattern, positional
heuristic, structural) are implemented. Strategy 5 (LLM fallback) is
deliberately not built in this version — per user instruction, this codebase
has no LLM dependency. A field that clears no strategy's confidence threshold
resolves to Strategy 6: `value: null` with a `review_reason`, which is a
valid, documented outcome, not a failure.
"""
from __future__ import annotations

import re
from typing import Optional

from .normalize import parse_currency_id, parse_date_id, parse_number_words_id, parse_rate
from .schema import value_object

LABEL_DICTIONARIES = {
    "contract_number": ["Nomor Kontrak", "Nomor Surat Perjanjian", "Nomor SPK", "Nomor Perjanjian", "Nomor Dokumen", "Nomor", "No."],
    "contract_name": ["Nama Pekerjaan", "Paket Pekerjaan", "Sub Kegiatan", "Nama Paket", "Objek Perjanjian", "Pekerjaan", "Kegiatan", "Perihal", "Tentang"],
    "value": ["Nilai Perjanjian", "Nilai Kontrak", "Harga Kontrak", "Nilai Pekerjaan", "Pagu Anggaran"],
}

DOCUMENT_TYPE_SIGNALS = [
    ("kontrak_konstruksi", "Surat Perjanjian (Kontrak) Kerja Konstruksi", [r"KONTRAK\s+KERJA\s+KONSTRUKSI", r"SURAT\s+PERJANJIAN"]),
    ("kontrak_pengadaan_barang", "Surat Perjanjian Pengadaan Barang", [r"PENGADAAN\s+BARANG"]),
    ("kontrak_jasa_konsultansi", "Surat Perjanjian Jasa Konsultansi", [r"JASA\s+KONSULTANSI"]),
    ("surat_perintah_kerja", "Surat Perintah Kerja", [r"\bSPK\b", r"SURAT\s+PERINTAH\s+KERJA"]),
]
SUBTYPE_SIGNALS = [
    ("kontrak_harga_satuan", [r"KONTRAK\s+HARGA\s+SATUAN", r"HARGA\s+SATUAN"]),
    ("kontrak_lump_sum", [r"LUMP\s*SUM"]),
]


def _label_lookup(full_text: str, labels: list[str], value_re: str = r"[^\n]{1,150}", require_colon: bool = False) -> list[dict]:
    """Strategy 1. Returns candidates ordered by label specificity (dict order).

    `require_colon` matters for short, generic labels like "Nomor"/"No.":
    without it, an Indonesian bracketed drafting instruction like "[diisi
    nomor Kontrak]" or "[diisi nomor faksimili Penyedia]" — ordinary prose
    that just happens to contain the word "nomor" with no colon anywhere
    near it — is indistinguishable from a genuine "Nomor : <value>" label,
    and wins the value "Kontrak" or "faksimili" as a fake contract number.
    A real label:value pair in this document family always has the colon;
    prose mentioning the label word in passing does not. Longer, more
    specific labels ("Nama Pekerjaan", "Paket Pekerjaan") don't need this —
    they're not common enough in ordinary prose to false-match, and some of
    their genuine title-block instances have no colon at all (the value
    sits on the next line instead), so requiring one there would break
    contract_name extraction instead of fixing anything."""
    colon_part = r":\s*" if require_colon else r":?\s*"
    candidates = []
    for rank, label in enumerate(labels):
        pattern = re.compile(rf"\b{re.escape(label)}\s*{colon_part}({value_re})", re.IGNORECASE)
        for m in pattern.finditer(full_text):
            raw_value = m.group(1).strip(" \t.-")
            if not raw_value:
                continue
            candidates.append(
                {
                    "value_raw": raw_value,
                    "label_matched": label,
                    "specificity_rank": rank,
                    "start": m.start(1),
                    "end": m.end(1),
                    "confidence": max(0.90, 0.99 - rank * 0.02),
                }
            )
    return candidates


def _score_and_pick(candidates: list[dict], occurrence_counts: Optional[dict] = None) -> tuple[Optional[dict], list[dict]]:
    if not candidates:
        return None, []
    scored = []
    for c in candidates:
        score = c["confidence"]
        score *= 1.0 - c.get("specificity_rank", 0) * 0.03
        if occurrence_counts and c["value_raw"] in occurrence_counts and occurrence_counts[c["value_raw"]] >= 2:
            score *= 1.15
        c = dict(c, score=min(1.0, score))
        scored.append(c)
    scored.sort(key=lambda c: -c["score"])
    best = scored[0]
    ambiguous = len(scored) > 1 and (best["score"] - scored[1]["score"]) < 0.15
    if ambiguous:
        best = dict(best, flags=["ambiguous"])
    return best, scored[1:6]


def resolve_document_type(full_text: str) -> dict:
    best_type, best_label, best_hits = "unknown", None, 0
    for type_id, label, patterns in DOCUMENT_TYPE_SIGNALS:
        hits = sum(1 for p in patterns if re.search(p, full_text, re.IGNORECASE))
        if hits > best_hits:
            best_type, best_label, best_hits = type_id, label, hits

    subtype = None
    for subtype_id, patterns in SUBTYPE_SIGNALS:
        if any(re.search(p, full_text, re.IGNORECASE) for p in patterns):
            subtype = subtype_id
            break

    if best_hits == 0:
        return value_object(value="unknown", confidence=0.0, method="positional", flags=["no_title_signal_matched"])

    confidence = min(0.97, 0.6 + 0.15 * best_hits)
    return value_object(
        value=best_type,
        raw=best_label,
        confidence=confidence,
        method="contextual_pattern",
        label_id=best_label,
        subtype=subtype,
    )


_GENERIC_QUALIFIER_TERMS = {"konstruksi", "pengadaan barang", "jasa konsultansi", "jasa lainnya", "barang", "jasa"}
_NEXT_LINE_STOP_RE = re.compile(r"^(Nomor|Nama|Tanggal|Jenis|Lokasi|Sumber)\b", re.IGNORECASE)


def _extend_title_block_value(full_text: str, candidate: dict) -> str:
    """Title blocks often print the label and a generic qualifier
    ("Paket Pekerjaan Konstruksi") on one line and the actual specific name
    on the next, with no colon to delimit it. A short or generic same-line
    value is extended with the following non-empty line."""
    value = candidate["value_raw"]
    is_generic = value.strip().lower() in _GENERIC_QUALIFIER_TERMS
    if len(value) > 20 and not is_generic:
        return value
    tail_lines = [ln.strip() for ln in full_text[candidate["end"]: candidate["end"] + 200].split("\n") if ln.strip()]
    if tail_lines and not _NEXT_LINE_STOP_RE.match(tail_lines[0]):
        # A purely generic qualifier ("Konstruksi") on the label line is not
        # part of the project name — the next line replaces it rather than
        # being appended to it. A short-but-specific value is kept as a prefix.
        return tail_lines[0] if is_generic else f"{value} {tail_lines[0]}".strip()
    return value


def resolve_contract_name(full_text: str) -> dict:
    candidates = _label_lookup(full_text, LABEL_DICTIONARIES["contract_name"])
    best, rest = _score_and_pick(candidates)
    if not best:
        return value_object(confidence=0.0, method="unresolved", flags=["review_required"])
    extended_value = _extend_title_block_value(full_text, best)
    return value_object(
        value=extended_value.title() if extended_value.isupper() else extended_value,
        raw=best["value_raw"],
        confidence=best["score"],
        method="regex_labeled",
        label_matched=best["label_matched"],
        evidence={"char_span": [best["start"], best["end"]]},
        candidates=[{"value": c["value_raw"], "label_matched": c["label_matched"], "score": round(c["score"], 2)} for c in rest],
        flags=best.get("flags", []),
    )


_PLACEHOLDER_DOTS_RE = re.compile(r"\.{3,}")
_CITATION_TENTANG_RE = re.compile(r"^\s*tentang\b", re.IGNORECASE)


def resolve_contract_number(full_text: str) -> dict:
    value_re = r"[A-Z0-9][A-Z0-9./\-]{4,60}"
    candidates = _label_lookup(full_text, LABEL_DICTIONARIES["contract_number"], value_re=value_re, require_colon=True)
    # "Nomor : X tentang Y" is the standard Indonesian legal-citation shape
    # ("Undang-Undang No. 2 Tahun 2017 tentang Jasa Konstruksi", "Surat
    # Edaran ... Nomor: HK.02.02/II/753/2020 tentang Revisi ke-3 ...") — every
    # law/decree/circular referenced in the recitals is cited this way. The
    # contract's OWN number is never followed by "tentang"; requiring the
    # colon (above) filters out bracket-instruction false matches but does
    # nothing against this one, since real citations genuinely have a colon
    # too. Filtering by what follows the value, not just what precedes it.
    candidates = [c for c in candidates if not _CITATION_TENTANG_RE.match(full_text[c["end"]: c["end"] + 15])]
    occurrence_counts = {}
    for c in candidates:
        occurrence_counts[c["value_raw"]] = full_text.count(c["value_raw"])
    best, rest = _score_and_pick(candidates, occurrence_counts)
    if not best:
        return value_object(confidence=0.0, method="unresolved", flags=["review_required"])
    if _PLACEHOLDER_DOTS_RE.search(best["value_raw"]):
        # A cover-sheet number like "602.1/.../SP-KONT/CK.AG/PUPR/.../2021"
        # has real segments filled in but the sequence/date segments are
        # still "..." blanks — the whole thing is a template, not an
        # assigned number, the same way an all-blank field is.
        return value_object(confidence=0.0, method="unresolved", flags=["template_placeholder", "review_required"])
    return value_object(
        value=best["value_raw"],
        raw=best["value_raw"],
        confidence=min(0.99, best["score"]),
        method="regex_labeled",
        occurrence_count=occurrence_counts.get(best["value_raw"], 1),
        evidence={"char_span": [best["start"], best["end"]]},
        candidates=[{"value": c["value_raw"], "score": round(c["score"], 2)} for c in rest],
        flags=best.get("flags", []),
    )


_ROLE_MARKERS = [
    (r"PIHAK\s+PERTAMA", "pihak_pertama"),
    (r"PIHAK\s+KEDUA", "pihak_kedua"),
]
_NIP_RE = re.compile(r"NIP\.?\s*[:.]?\s*(\d[\d\s]{10,25}\d)")
_REPRESENTATIVE_NAME_RE = re.compile(r"\n([A-Z][A-Za-zÀ-ÿ.,'\- ]{2,60}(?:,\s*[A-Z]{1,6}(?:\.[A-Za-z]{1,6})*)?)\s*\n(?=[^\n]{0,40}NIP)")
DISEBUT_ROLE_RE = re.compile(r'selanjutnya\s+disebut\s+["“]([^"”]{1,40})["”]', re.IGNORECASE)
# Terms that "selanjutnya disebut" also commonly defines but that are not
# parties (the agreement itself, its amendments, etc.) — excluded rather than
# allow-listed, since party role vocabulary otherwise varies a lot across
# contract types (Penyedia/Kontraktor, PPKom/Pemberi Kerja, ...).
_SELF_REFERENCE_TERMS = {
    "kontrak", "perjanjian", "spmk", "adendum", "amandemen", "dokumen kontrak", "spk",
    "pekerjaan konstruksi",
}
_ORG_BEFORE_DISEBUT_RE = re.compile(r"atas\s+nama\s+(.+?)\s*(?:,\s*)?selanjutnya\s+disebut", re.IGNORECASE | re.DOTALL)
_NAME_LABEL_RE = re.compile(r"\bNama\s*:?\s*([^\n]{2,80})", re.IGNORECASE)
_POSITION_LABEL_RE = re.compile(r"\bJabatan\s*:?\s*([^\n]{2,80})", re.IGNORECASE)
_ADDRESS_LABEL_RE = re.compile(r"\bBerkedudukan\s+di\s*:?\s*([^\n]{2,150})", re.IGNORECASE)
_PLACEHOLDER_VALUE_RE = re.compile(r"…|\.{3,}|\[")


_ABBREVIATION_TAIL_RE = re.compile(r"[A-Za-z]\.[A-Za-z]{1,4}\.$")


def _clean_window_value(raw: str) -> tuple[str | None, bool]:
    """Strips a captured label value, returns (value_or_None, is_placeholder)."""
    raw = re.sub(r"\s+", " ", raw).strip()
    # A trailing "." is normally junk line-end punctuation and stripped —
    # except when it's the closing dot of a multi-part abbreviation
    # ("S.T.", "M.T.", "S.Pi."), which blind stripping silently corrupts
    # (confirmed on a real name: "INDRARTO WIDYATMOKO, S.T., M.T." would
    # otherwise lose its final period).
    if raw.endswith(".") and not _ABBREVIATION_TAIL_RE.search(raw):
        raw = raw[:-1]
    raw = raw.strip(" \t:")
    if not raw:
        return None, True
    is_placeholder = bool(_PLACEHOLDER_VALUE_RE.search(raw))
    return (None if is_placeholder else raw), is_placeholder


_ORG_TRIM_RE = re.compile(r"\s+(berdasarkan|yang\s+beralamat|yang\s+berkedudukan)\b", re.IGNORECASE)


_PARTY_WINDOW_BACK = 950


def _extract_party_from_disebut(full_text: str, m: re.Match, role_label: str, party_id: str, prev_boundary: int) -> dict:
    # 950, not 600: a party's "Nama :" label can sit well over 600 characters
    # before its own "selanjutnya disebut" clause whenever a long-winded SK/
    # decree citation comes between them (measured up to 714 chars across the
    # ground-truth specimens) — a shorter window silently missed the name
    # despite it appearing verbatim, even though the adjacent NIP (matched by
    # a different, unrelated regex closer to the disebut clause) resolved
    # fine. Floored at prev_boundary, same as the org search below, so a
    # wider window can't reach back far enough to grab the PREVIOUS party's
    # own Nama/Jabatan/NIP when two parties sit close together.
    window_start = max(0, m.start() - _PARTY_WINDOW_BACK, prev_boundary)
    window = full_text[window_start: min(len(full_text), m.end() + 400)]

    # Search only back to the previous party's disebut clause (or 950 chars,
    # whichever is closer) and take the LAST match in that span — otherwise a
    # non-greedy search from further back can jump past this party's own
    # placeholder text and re-match the previous party's "atas nama ...
    # selanjutnya disebut" clause when the two are close together.
    org_search_start = max(0, m.start() - _PARTY_WINDOW_BACK, prev_boundary)
    org_matches = list(_ORG_BEFORE_DISEBUT_RE.finditer(full_text[org_search_start: m.end()]))
    org_value, org_placeholder = (None, True)
    if org_matches:
        org_raw = org_matches[-1].group(1)
        org_raw = _ORG_TRIM_RE.split(org_raw)[0]
        org_value, org_placeholder = _clean_window_value(org_raw)

    nip_m = _NIP_RE.search(window)
    name_m = _NAME_LABEL_RE.search(window)
    position_m = _POSITION_LABEL_RE.search(window)
    address_m = _ADDRESS_LABEL_RE.search(window)

    name_value, name_placeholder = (None, True)
    if name_m:
        name_value, name_placeholder = _clean_window_value(name_m.group(1))
    position_value, _ = _clean_window_value(position_m.group(1)) if position_m else (None, True)
    address_value, _ = _clean_window_value(address_m.group(1)) if address_m else (None, True)

    flags = []
    if org_placeholder:
        flags.append("template_placeholder")
    if name_placeholder:
        flags.append("representative_unresolved")

    return {
        "party_id": party_id,
        "role": re.sub(r"\s+", "_", role_label.strip().lower()) or party_id,
        "role_label": role_label.strip(),
        "organization": {"value": org_value, "type": "unknown", "confidence": 0.85 if org_value else 0.0},
        "representative": {
            "name": name_value,
            "position": position_value,
            "identifier": {"type": "NIP", "value": re.sub(r"\s+", " ", nip_m.group(1)).strip()} if nip_m else None,
            "confidence": 0.8 if name_value else 0.0,
        },
        "address": address_value,
        "flags": flags,
    }


def resolve_parties(full_text: str) -> dict:
    """Strategy 4 (structural). Two window strategies, tried in order:
    explicit `PIHAK PERTAMA` / `PIHAK KEDUA` markers (older contract style),
    falling back to `... yang bertindak untuk dan atas nama X, selanjutnya
    disebut "Y"` definitions (the style this Perpres 16/2018 template uses),
    which is also where organization/representative/NIP actually live."""
    parties = []
    marker_positions = [(re.search(p, full_text, re.IGNORECASE), role) for p, role in _ROLE_MARKERS]
    marker_positions = [(m, role) for m, role in marker_positions if m]

    disebut_matches = [
        m for m in DISEBUT_ROLE_RE.finditer(full_text)
        # Whitespace-normalized: a quoted role that line-wraps mid-phrase
        # ('disebut "Pekerjaan\nKonstruksi"') must still compare equal to its
        # single-space form in _SELF_REFERENCE_TERMS.
        if re.sub(r"\s+", " ", m.group(1)).strip().lower() not in _SELF_REFERENCE_TERMS
    ]

    if marker_positions:
        marker_positions.sort(key=lambda t: t[0].start())
        for idx, (m, role) in enumerate(marker_positions):
            window_end = marker_positions[idx + 1][0].start() if idx + 1 < len(marker_positions) else min(len(full_text), m.start() + 1200)
            window = full_text[m.start():window_end]

            nip_m = _NIP_RE.search(window)
            rep_m = _REPRESENTATIVE_NAME_RE.search(window)
            role_label_m = re.search(r'["“]([^"”]{1,30})["”]', window)

            party = {
                "party_id": f"party_{idx + 1}",
                "role": role,
                "role_label": role_label_m.group(1) if role_label_m else None,
                "organization": {"value": None, "type": "unknown", "confidence": 0.0},
                "representative": {
                    "name": rep_m.group(1).strip() if rep_m else None,
                    "position": None,
                    "identifier": {"type": "NIP", "value": re.sub(r"\s+", " ", nip_m.group(1)).strip()} if nip_m else None,
                    "confidence": 0.7 if rep_m else 0.0,
                },
                "flags": [] if (nip_m or rep_m) else ["template_placeholder"],
            }
            parties.append(party)
    elif disebut_matches:
        prev_boundary = 0
        for idx, m in enumerate(disebut_matches[:6]):
            parties.append(_extract_party_from_disebut(full_text, m, m.group(1), f"party_{idx + 1}", prev_boundary))
            prev_boundary = m.end()

    if not parties:
        return value_object(value=[], confidence=0.0, method="unresolved", flags=["no_parties_detected", "review_required"])

    populated = sum(1 for p in parties if p["representative"]["name"] or p["organization"]["value"])
    confidence = round(0.4 + 0.3 * (populated / max(1, len(parties))), 2)
    flags = [] if populated == len(parties) else ["incomplete_parties"]
    return value_object(value=parties, confidence=confidence, method="structural", flags=flags)


_DATE_CONTEXT_LABELS = [
    (r"TAHUN\s+ANGGARAN", "fiscal_year"),
    (r"ditetapkan\s+di", "authority_date"),
    (r"[Tt]anggal\s*:?", "unlabeled_date"),
]
_DATE_SCAN_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-zé]+\s+\d{4})|(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})|(TAHUN\s+ANGGARAN\s+\d{4})",
    re.IGNORECASE,
)


def resolve_key_dates(full_text: str) -> dict:
    found = []
    seen_spans = set()
    for m in _DATE_SCAN_RE.finditer(full_text):
        raw = m.group(0)
        if (m.start(), m.end()) in seen_spans:
            continue
        seen_spans.add((m.start(), m.end()))
        window = full_text[max(0, m.start() - 60): m.start()]
        date_type = "unclassified_date"
        for pattern, dtype in _DATE_CONTEXT_LABELS:
            if re.search(pattern, window, re.IGNORECASE):
                date_type = dtype
                break
        iso, precision = parse_date_id(raw)
        is_placeholder = bool(re.search(r"…|\.{3,}|\[", window[-20:]))
        found.append(
            {
                "type": date_type,
                "date": iso,
                "raw": raw,
                "precision": precision,
                "confidence": 0.9 if iso else 0.0,
                "flags": ["template_placeholder"] if is_placeholder or iso is None else [],
            }
        )
    if not found:
        return value_object(value=[], confidence=0.0, method="unresolved", flags=["no_dates_detected"])
    unresolved = sum(1 for d in found if d["date"] is None)
    confidence = round(1.0 - (unresolved / len(found)) * 0.6, 2)
    flags = ["multiple_dates_unresolved"] if unresolved else []
    return value_object(value=found, confidence=confidence, method="contextual_pattern", flags=flags)


# "kalende(?:r)?" — not just "kalender" — because the word is genuinely
# truncated at a page break in this document ("...hari kalende\nDengan...").
# A regex that insists on the full spelling silently drops that occurrence.
_DURATION_RE = re.compile(r"(\d{1,4})\s*\(([^)]{2,60})\)\s*hari\s*kalende(?:r)?\b", re.IGNORECASE)
_DURATION_SUBTYPE_KEYWORDS = [
    (re.compile(r"masa\s+pelaksanaan", re.IGNORECASE), "masa_pelaksanaan"),
    (re.compile(r"masa\s+pemeliharaan", re.IGNORECASE), "masa_pemeliharaan"),
]
_VALUE_RE = re.compile(r"Rp\.?\s*([\d.,]+|\.{3,})", re.IGNORECASE)
# Generic monetary-amount scan, separate from the labeled contract_value
# lookup above. Requires a digit immediately after "Rp" (with optional dot/
# space), which is what keeps it from matching the blank-template placeholder
# ("Rp. .................." has no digit there — only dots).
_MONETARY_RE = re.compile(r"Rp\.?\s*(\d[\d.,]*)", re.IGNORECASE)
_MONETARY_SUBTYPE_KEYWORDS = [
    (re.compile(r"meterai|materai", re.IGNORECASE), "meterai"),
]
# Indonesian contract prose wraps every ~10 words, so the rate itself is
# often on the line after "denda" — exclude only sentence-terminal periods
# from the window, not newlines.
_PENALTY_CONTEXT_RE = re.compile(r"denda[^.]{0,150}", re.IGNORECASE)
_DPPA_RE = re.compile(r"\b\d{1,2}(?:\.\d{1,2}){4,6}\b")


def _classify_by_nearby_keyword(full_text: str, match_start: int, keyword_table: list, window: int = 250) -> str:
    """Scans backward from a match for the closest preceding keyword,
    returning its label, or 'unclassified' if none is found in range."""
    context = full_text[max(0, match_start - window): match_start]
    best_label, best_pos = "unclassified", -1
    for pattern, label in keyword_table:
        for m in pattern.finditer(context):
            if m.start() > best_pos:
                best_pos, best_label = m.start(), label
    return best_label


def resolve_key_numbers(full_text: str) -> dict:
    numbers = []

    # Multiple, distinct duration figures can legitimately coexist (e.g.
    # "masa pelaksanaan" vs "masa pemeliharaan") — .search()-ing for only the
    # first one silently drops every duration after it.
    for dur_m in _DURATION_RE.finditer(full_text):
        amount = int(dur_m.group(1))
        words_value = parse_number_words_id(dur_m.group(2))
        subtype = _classify_by_nearby_keyword(full_text, dur_m.start(), _DURATION_SUBTYPE_KEYWORDS)
        numbers.append(
            {
                "type": "duration",
                "subtype": subtype,
                "amount": amount,
                "unit": "hari_kalender",
                "raw": dur_m.group(0),
                "confidence": 0.95,
                "words_check": "passed" if words_value == amount else "mismatch",
            }
        )

    value_candidates = _label_lookup(full_text, LABEL_DICTIONARIES["value"], value_re=r"[^\n]{1,60}")
    best_value, _ = _score_and_pick(value_candidates)
    contract_value_span = None
    if best_value:
        is_placeholder = bool(re.search(r"\.{3,}|…", best_value["value_raw"]))
        amount = None if is_placeholder else parse_currency_id(best_value["value_raw"])
        if amount is not None:
            contract_value_span = (best_value["start"], best_value["end"])
        numbers.append(
            {
                "type": "contract_value",
                "amount": amount,
                "currency": "IDR",
                "raw": best_value["value_raw"],
                "confidence": 0.0 if amount is None else 0.9,
                "flags": ["template_placeholder"] if amount is None else [],
            }
        )

    # Generic monetary amounts NOT already claimed by the labeled
    # contract_value lookup above — e.g. stamp-duty (meterai) boilerplate.
    # Deliberately kept as its own `monetary` type rather than folded into
    # contract_value: a currency scan that just takes the first or largest
    # Rp figure on the page would wrongly promote this kind of incidental
    # amount to the contract value.
    seen_monetary = set()
    for mon_m in _MONETARY_RE.finditer(full_text):
        if contract_value_span and contract_value_span[0] <= mon_m.start() < contract_value_span[1]:
            continue
        amount = parse_currency_id(mon_m.group(1))
        if amount is None:
            continue
        subtype = _classify_by_nearby_keyword(full_text, mon_m.start(), _MONETARY_SUBTYPE_KEYWORDS)
        dedupe_key = (subtype, amount)
        if dedupe_key in seen_monetary:
            continue
        seen_monetary.add(dedupe_key)
        numbers.append(
            {
                "type": "monetary",
                "subtype": subtype,
                "amount": amount,
                "currency": "IDR",
                "raw": mon_m.group(0),
                "confidence": 0.75 if subtype != "unclassified" else 0.5,
            }
        )

    # Distinct penalty rates coexist under one document (delay penalty vs.
    # quality-defect penalty, sometimes at the same 1/1000 rate) — .search()
    # for the first "denda...rate" pairing collapses all of them into one.
    # The qualifying phrase ("keterlambatan" / "cacat mutu") sits AFTER
    # "denda" within the match itself, not before it, so subtype is
    # classified from the match text, not a backward-looking window.
    seen_penalty = set()
    for penalty_m in _PENALTY_CONTEXT_RE.finditer(full_text):
        rate = parse_rate(penalty_m.group(0))
        if rate is None:
            continue
        match_text = penalty_m.group(0)
        if re.search(r"cacat\s+mutu", match_text, re.IGNORECASE):
            subtype = "denda_cacat_mutu"
        elif re.search(r"keterlambatan", match_text, re.IGNORECASE):
            subtype = "denda_keterlambatan"
        else:
            subtype = "unclassified"
        dedupe_key = (subtype, round(rate, 6))
        if dedupe_key in seen_penalty:
            continue
        seen_penalty.add(dedupe_key)
        numbers.append(
            {
                "type": "penalty_rate",
                "subtype": subtype,
                "amount": rate,
                "unit": "ratio",
                "raw": match_text.strip(),
                "confidence": 0.85,
            }
        )

    dppa_m = _DPPA_RE.search(full_text)
    if dppa_m:
        numbers.append(
            {
                "type": "reference_number",
                "subtype": "DPPA-SKPD",
                "value": dppa_m.group(0),
                "raw": dppa_m.group(0),
                "confidence": 0.8,
            }
        )

    if not numbers:
        return value_object(value=[], confidence=0.0, method="unresolved", flags=["no_numbers_detected"])
    missing_value = any(n["type"] == "contract_value" and n.get("amount") is None for n in numbers)
    confidence = round(sum(n["confidence"] for n in numbers) / len(numbers), 2)
    return value_object(value=numbers, confidence=confidence, method="contextual_pattern", flags=["contract_value_missing"] if missing_value else [])


def resolve_core(full_text: str, document_status: str) -> dict:
    document_type = resolve_document_type(full_text)
    contract_name = resolve_contract_name(full_text)
    contract_number = resolve_contract_number(full_text)
    parties = resolve_parties(full_text)
    key_dates = resolve_key_dates(full_text)
    key_numbers = resolve_key_numbers(full_text)

    fields = {
        "document_type": document_type,
        "contract_name": contract_name,
        "contract_number": contract_number,
        "parties": parties,
        "key_dates": key_dates,
        "key_numbers": key_numbers,
    }
    populated = sum(1 for f in fields.values() if f["value"] not in (None, [], "unknown"))
    overall_confidence = round(sum(f["confidence"] for f in fields.values()) / len(fields), 3)
    review_reasons = sorted({flag for f in fields.values() for flag in f["flags"]})

    fields["_status"] = {
        "document_status": document_status,
        "fields_populated": populated,
        "fields_null": len(fields) - populated,
        "overall_confidence": overall_confidence,
        "requires_human_review": overall_confidence < 0.6 or bool(review_reasons),
        "review_reasons": review_reasons,
    }
    return fields
