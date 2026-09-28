"""Stage 8 — Core Field Resolution. """
from __future__ import annotations

import re
from typing import Optional

from vocabulary import Vocabulary, for_profile

from .field_context import FieldContext
from .normalize import MONTHS_ID, parse_currency_id, parse_date_id, parse_number_words_id, parse_rate
from .schema import value_object



def default_vocabulary() -> Vocabulary:
    return for_profile(None)




def _label_lookup(full_text: str, labels: list[str], value_re: str = r"[^\n]{1,150}",
                  require_colon: bool = False, line_start_only: bool = False) -> list[dict]:
    colon_part = r":\s*" if require_colon else r":?\s*"
    prefix = r"(?m)^[\s|]*" if line_start_only else r"\b"
    candidates = []
    for rank, label in enumerate(labels):
        pattern = re.compile(rf"{prefix}{re.escape(label)}\s*{colon_part}({value_re})", re.IGNORECASE)
        for m in pattern.finditer(full_text):
            raw_value = m.group(1).strip(" \t.-")
            if not raw_value:
                continue
            if line_start_only and not (raw_value[0].isupper() or raw_value[0].isdigit()
                                        or raw_value[0] in ".…["):
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


TITLE_REGION_BOOST = 1.15
OUTSIDE_TITLE_PENALTY = 0.75
GENERIC_LABEL_RANK = 4
GENERIC_OUTSIDE_TITLE_CAP = 0.55


def _score_and_pick(candidates: list[dict], occurrence_counts: Optional[dict] = None,
                    context: FieldContext | None = None) -> tuple[Optional[dict], list[dict]]:
    if not candidates:
        return None, []
    scored = []
    for c in candidates:
        score = c["confidence"]
        score *= 1.0 - c.get("specificity_rank", 0) * 0.03
        if occurrence_counts and c["value_raw"] in occurrence_counts and occurrence_counts[c["value_raw"]] >= 2:
            score *= 1.15
        if context is not None:
            first_segment = context.first_sub_document()
            segment = context.sub_document_at(c["start"])
            in_opening_part = first_segment is None or segment == first_segment
            in_title = context.in_title_region(c["start"]) and in_opening_part
            score *= TITLE_REGION_BOOST if in_title else OUTSIDE_TITLE_PENALTY
            if not in_title and c.get("specificity_rank", 0) >= GENERIC_LABEL_RANK:
                score = min(score, GENERIC_OUTSIDE_TITLE_CAP)
            if not in_opening_part:
                score = min(score, GENERIC_OUTSIDE_TITLE_CAP)
        c = dict(c, score=min(1.0, score))
        scored.append(c)
    scored.sort(key=lambda c: -c["score"])
    best = scored[0]
    ambiguous = len(scored) > 1 and (best["score"] - scored[1]["score"]) < 0.15
    if ambiguous:
        best = dict(best, flags=["ambiguous"])
    return best, scored[1:6]


TITLE_REGION_WEIGHT = 3.0
BODY_WEIGHT = 1.0
GENERIC_PATTERN_WEIGHT = 0.5


def _signal_score(patterns: list[str], full_text: str, context: FieldContext | None, weight: float) -> float:
    score = 0.0
    for pattern in patterns:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            if context is None:
                score += weight
                break
            score += weight * (TITLE_REGION_WEIGHT if context.in_title_region(match.start()) else BODY_WEIGHT)
    return score


def resolve_document_type(full_text: str, vocab: Vocabulary | None = None,
                          context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    best_type, best_label, best_hits = "unknown", None, 0.0
    for signal in vocab.get("document_type_signals") or []:
        specific = list(signal["patterns"])
        generic = list(signal.get("generic_patterns") or [])
        if context is None:
            hits = float(sum(1 for p in specific + generic if re.search(p, full_text, re.IGNORECASE)))
        else:
            hits = (_signal_score(specific, full_text, context, 1.0)
                    + _signal_score(generic, full_text, context, GENERIC_PATTERN_WEIGHT))
        if hits > best_hits:
            best_type, best_label, best_hits = signal["id"], signal.get("label"), hits

    subtype = None
    for signal in vocab.get("subtype_signals") or []:
        if any(re.search(p, full_text, re.IGNORECASE) for p in signal["patterns"]):
            subtype = signal["id"]
            break

    if best_hits == 0:
        return value_object(value="unknown", confidence=0.0, method="positional", flags=["no_title_signal_matched"])

    confidence = min(0.97, 0.6 + 0.15 * min(best_hits, 3.0))
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
    value = candidate["value_raw"]
    is_generic = value.strip().lower() in _GENERIC_QUALIFIER_TERMS
    if len(value) > 20 and not is_generic:
        return value
    tail_lines = [ln.strip() for ln in full_text[candidate["end"]: candidate["end"] + 200].split("\n") if ln.strip()]
    if tail_lines and not _NEXT_LINE_STOP_RE.match(tail_lines[0]):
        return tail_lines[0] if is_generic else f"{value} {tail_lines[0]}".strip()
    return value


def resolve_contract_name(full_text: str, vocab: Vocabulary | None = None,
                          context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    candidates = _label_lookup(full_text, vocab.labels("contract_name"), line_start_only=context is not None)
    best, rest = _score_and_pick(candidates, context=context)
    if not best:
        return value_object(confidence=0.0, method="unresolved", flags=["review_required"])
    extended_value = _extend_title_block_value(full_text, best)
    if context is not None and _PLACEHOLDER_VALUE_RE.search(extended_value):
        return value_object(confidence=0.0, method="unresolved",
                            raw=best["value_raw"],
                            flags=["template_placeholder", "review_required"])
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


def resolve_contract_number(full_text: str, vocab: Vocabulary | None = None,
                            context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    value_re = r"[A-Z0-9][A-Z0-9./\-]{4,60}"
    candidates = _label_lookup(full_text, vocab.labels("contract_number"), value_re=value_re,
                               require_colon=True, line_start_only=context is not None)
    candidates = [c for c in candidates if not _CITATION_TENTANG_RE.match(full_text[c["end"]: c["end"] + 15])]
    occurrence_counts = {}
    for c in candidates:
        occurrence_counts[c["value_raw"]] = full_text.count(c["value_raw"])
    best, rest = _score_and_pick(candidates, occurrence_counts, context=context)
    if not best:
        return value_object(confidence=0.0, method="unresolved", flags=["review_required"])
    if _PLACEHOLDER_DOTS_RE.search(best["value_raw"]):
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
    raw = re.sub(r"\s+", " ", raw).strip()
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
    window_start = max(0, m.start() - _PARTY_WINDOW_BACK, prev_boundary)
    window = full_text[window_start: min(len(full_text), m.end() + 400)]

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


def resolve_parties(full_text: str, vocab: Vocabulary | None = None,
                    context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    role_markers = [(m["pattern"], m["role"]) for m in vocab.get("party_role_markers") or []]
    self_reference_terms = {t.lower() for t in vocab.get("self_reference_terms") or []}
    parties = []
    marker_positions = [(re.search(p, full_text, re.IGNORECASE), role) for p, role in role_markers]
    marker_positions = [(m, role) for m, role in marker_positions if m]

    disebut_matches = [
        m for m in DISEBUT_ROLE_RE.finditer(full_text)
        if re.sub(r"\s+", " ", m.group(1)).strip().lower() not in self_reference_terms
    ]

    if marker_positions:
        marker_positions.sort(key=lambda t: t[0].start())
        for idx, (m, role) in enumerate(marker_positions):
            window_end = marker_positions[idx + 1][0].start() if idx + 1 < len(marker_positions) else min(len(full_text), m.start() + 1200)
            window = full_text[m.start():window_end]

            nip_m = _NIP_RE.search(window)
            rep_m = _REPRESENTATIVE_NAME_RE.search(window)
            role_label_m = re.search(r'["“]([^"”]{1,30})["”]', window)

            name_m = _NAME_LABEL_RE.search(window)
            position_m = _POSITION_LABEL_RE.search(window)
            address_m = _ADDRESS_LABEL_RE.search(window)
            labelled_name, name_is_placeholder = _clean_window_value(name_m.group(1)) if name_m else (None, True)
            position_value, _ = _clean_window_value(position_m.group(1)) if position_m else (None, True)
            address_value, _ = _clean_window_value(address_m.group(1)) if address_m else (None, True)

            name_value = labelled_name or (rep_m.group(1).strip() if rep_m else None)
            name_confidence = 0.85 if labelled_name else (0.7 if rep_m else 0.0)

            party = {
                "party_id": f"party_{idx + 1}",
                "role": role,
                "role_label": role_label_m.group(1) if role_label_m else None,
                "organization": {"value": None, "type": "unknown", "confidence": 0.0},
                "representative": {
                    "name": name_value,
                    "position": position_value,
                    "identifier": {"type": "NIP", "value": re.sub(r"\s+", " ", nip_m.group(1)).strip()} if nip_m else None,
                    "confidence": name_confidence,
                },
                "address": address_value,
                "flags": [] if name_value else (["template_placeholder"] if name_is_placeholder else ["representative_unresolved"]),
            }
            parties.append(party)
    elif disebut_matches:
        prev_boundary = 0
        for idx, m in enumerate(disebut_matches[:6]):
            parties.append(_extract_party_from_disebut(full_text, m, m.group(1), f"party_{idx + 1}", prev_boundary))
            prev_boundary = m.end()

    if context is not None:
        parties = _collapse_repeated_parties(parties)

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
_MONTH_ALTERNATION = "|".join(sorted(MONTHS_ID, key=len, reverse=True))
_DATE_SCAN_RE = re.compile(
    rf"((?:0?[1-9]|[12]\d|3[01])\s+(?:{_MONTH_ALTERNATION})\s+(?:19|20)\d{{2}})"
    r"|(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})"
    r"|(TAHUN\s+ANGGARAN\s+\d{4})",
    re.IGNORECASE,
)


def _party_content(party: dict) -> int:
    representative = party.get("representative") or {}
    return sum(1 for v in (
        (party.get("organization") or {}).get("value"),
        representative.get("name"),
        representative.get("position"),
        (representative.get("identifier") or {}).get("value"),
    ) if v)


def _collapse_repeated_parties(parties: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    order: list[str] = []
    for party in parties:
        role = party.get("role") or party.get("party_id")
        if role not in best:
            best[role] = party
            order.append(role)
        elif _party_content(party) > _party_content(best[role]):
            best[role] = party
    out = []
    for index, role in enumerate(order, start=1):
        party = dict(best[role])
        party["party_id"] = f"party_{index}"
        out.append(party)
    return out


def resolve_key_dates(full_text: str, vocab: Vocabulary | None = None,
                      context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    date_context_labels = [(e["pattern"], e["type"]) for e in vocab.get("date_context_labels") or []]
    found = []
    seen_spans = set()
    for m in _DATE_SCAN_RE.finditer(full_text):
        raw = m.group(0)
        if (m.start(), m.end()) in seen_spans:
            continue
        seen_spans.add((m.start(), m.end()))
        window = full_text[max(0, m.start() - 60): m.start()]
        date_type = "unclassified_date"
        for pattern, dtype in date_context_labels:
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
    if context is not None and all(d["type"] == "unclassified_date" for d in found):
        confidence = min(confidence, UNCLASSIFIED_DATES_CONFIDENCE_CAP)
        flags.append("dates_unclassified")
    return value_object(value=found, confidence=confidence, method="contextual_pattern", flags=flags)


_DURATION_RE = re.compile(
    r"(\d{1,4})\s*(?:\(([^)]{2,60})\))?\s*"
    r"(hari\s*kalende(?:r)?|hari\s*kerja|hari|minggu|bulan|tahun)\b",
    re.IGNORECASE,
)
_DURATION_UNITS = {
    "hari kalender": "hari_kalender", "hari kalende": "hari_kalender",
    "hari kerja": "hari_kerja", "hari": "hari",
    "minggu": "minggu", "bulan": "bulan", "tahun": "tahun",
}


def _duration_unit(raw_unit: str) -> str:
    key = re.sub(r"\s+", " ", raw_unit).strip().lower()
    return _DURATION_UNITS.get(key, key.replace(" ", "_"))
_DURATION_SUBTYPE_KEYWORDS = [
    (re.compile(r"masa\s+pelaksanaan", re.IGNORECASE), "masa_pelaksanaan"),
    (re.compile(r"masa\s+pemeliharaan", re.IGNORECASE), "masa_pemeliharaan"),
]
_VALUE_RE = re.compile(r"Rp\.?\s*([\d.,]+|\.{3,})", re.IGNORECASE)
_MONETARY_RE = re.compile(r"Rp\.?\s*(\d[\d.,]*)", re.IGNORECASE)
_MONETARY_SUBTYPE_KEYWORDS = [
    (re.compile(r"meterai|materai", re.IGNORECASE), "meterai"),
]
_PENALTY_CONTEXT_RE = re.compile(r"denda[^.]{0,150}", re.IGNORECASE)
_DPPA_RE = re.compile(r"\b\d{1,2}(?:\.\d{1,2}){4,6}\b")
_DPPA_LABEL_WINDOW = 80
UNCLASSIFIED_DATES_CONFIDENCE_CAP = 0.55


def _classify_by_nearby_keyword(full_text: str, match_start: int, keyword_table: list, window: int = 250) -> str:
    context = full_text[max(0, match_start - window): match_start]
    best_label, best_pos = "unclassified", -1
    for pattern, label in keyword_table:
        for m in pattern.finditer(context):
            if m.start() > best_pos:
                best_pos, best_label = m.start(), label
    return best_label


def resolve_key_numbers(full_text: str, vocab: Vocabulary | None = None,
                        context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    numbers = []

    seen_duration = set()
    for dur_m in _DURATION_RE.finditer(full_text):
        amount = int(dur_m.group(1))
        unit = _duration_unit(dur_m.group(3))
        subtype = _classify_by_nearby_keyword(full_text, dur_m.start(), vocab.keyword_table("duration_subtypes"))
        if dur_m.group(2) is None and subtype == "unclassified":
            continue
        dedupe_key = (subtype, amount, unit)
        if dedupe_key in seen_duration:
            continue
        seen_duration.add(dedupe_key)
        words_value = parse_number_words_id(dur_m.group(2)) if dur_m.group(2) else None
        numbers.append(
            {
                "type": "duration",
                "subtype": subtype,
                "amount": amount,
                "unit": unit,
                "raw": dur_m.group(0),
                "confidence": 0.95 if dur_m.group(2) else 0.8,
                "words_check": ("passed" if words_value == amount else "mismatch") if words_value is not None else "no_words",
            }
        )

    value_candidates = _label_lookup(full_text, vocab.labels("value"), value_re=r"[^\n]{1,60}")
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

    seen_monetary = set()
    for mon_m in _MONETARY_RE.finditer(full_text):
        if contract_value_span and contract_value_span[0] <= mon_m.start() < contract_value_span[1]:
            continue
        amount = parse_currency_id(mon_m.group(1))
        if amount is None:
            continue
        subtype = _classify_by_nearby_keyword(full_text, mon_m.start(), vocab.keyword_table("monetary_subtypes"))
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

    seen_penalty = set()
    for penalty_m in _PENALTY_CONTEXT_RE.finditer(full_text):
        rate = parse_rate(penalty_m.group(0))
        if rate is None:
            continue
        match_text = penalty_m.group(0)
        subtype = next(
            (label for pattern, label in vocab.keyword_table("penalty_subtypes") if pattern.search(match_text)),
            "unclassified",
        )
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

    reference_label_re = re.compile("|".join(vocab.get("reference_number_labels") or ["$^"]), re.IGNORECASE)
    dppa_m = next(
        (m for m in _DPPA_RE.finditer(full_text)
         if reference_label_re.search(full_text[max(0, m.start() - _DPPA_LABEL_WINDOW): m.end() + _DPPA_LABEL_WINDOW])),
        None,
    )
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


def resolve_core(full_text: str, document_status: str, vocab: Vocabulary | None = None,
                 context: FieldContext | None = None) -> dict:
    vocab = vocab or default_vocabulary()
    document_type = resolve_document_type(full_text, vocab, context)
    contract_name = resolve_contract_name(full_text, vocab, context)
    contract_number = resolve_contract_number(full_text, vocab, context)
    parties = resolve_parties(full_text, vocab, context)
    key_dates = resolve_key_dates(full_text, vocab, context)
    key_numbers = resolve_key_numbers(full_text, vocab, context)

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
