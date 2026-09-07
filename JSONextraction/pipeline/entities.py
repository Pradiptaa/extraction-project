"""Stage 7 — ENTITY EXTRACTION. Regex + gazetteer cascade over the tree and
the full document text. No statistical NER: contract numbers, NIP, dates,
clause refs and the `disingkat` abbreviation pattern are all cleanly regex-able
and more reliable than a general-purpose NER for these forms (see
analisis_pipeline_kontrak.md D.2).
"""
from __future__ import annotations

import re

from .normalize import parse_rate
from .tree import Node

NEGATION_TERMS = {"tidak", "bukan", "tanpa", "belum", "jangan", "dilarang"}
DEONTIC_OBLIGATION = {"wajib", "harus"}
DEONTIC_PROHIBITION = {"dilarang"}
DEONTIC_PERMISSION = {"dapat", "boleh"}

CLAUSE_REF_RE = re.compile(
    r"\b[Pp]asal\s+(\d{1,3}(?:\.\d{1,3}){0,3})|"
    r"\bSSKK\s+(\d{1,3}(?:\.\d{1,3}){0,3}(?:\.[a-z])?)",
)
LEGAL_CITATION_RE = re.compile(
    r"\b(UU|PP|Perpres|Permen\w*|Keppres)\s+No\.?\s*(\d+)\s*(?:Tahun|/)\s*(\d{4})",
    re.IGNORECASE,
)
NIP_RE = re.compile(r"\bNIP\.?\s*[:.]?\s*(\d{8}\s?\d{6}\s?\d\s?\d{3})\b")
DPPA_RE = re.compile(r"\b\d{1,2}(?:\.\d{1,2}){4,6}\b")
DISINGKAT_RE = re.compile(
    r'([A-Za-z][A-Za-zÀ-ÿ .,\-]{2,80}?)\s+yang\s+selanjutnya\s+disingkat\s+["“]?([A-Z][A-Za-z./]{1,15})["”]?'
)
DISEBUT_RE = re.compile(
    r'([A-Za-zÀ-ÿ .,\-]{2,120}?)\s+selanjutnya\s+disebut\s+["“]([^"”]{1,60})["”]'
)
PLACEHOLDER_RE = re.compile(r"…{1,}|\.{4,}|\[[^\]]{1,80}\]")
PERCENT_PERMILLE_FRACTION_RE = re.compile(r"(\d+(?:[.,]\d+)?\s*[‰%])|(\d+\s*/\s*\d+)")


def build_label_index(nodes: list[Node]) -> dict[str, str]:
    """label_normalized -> node_id, for clause/subclause/article nodes only."""
    index: dict[str, str] = {}
    for n in nodes:
        if n.node_type in ("clause", "subclause", "article") and n.label_normalized:
            index[n.label_normalized] = n.node_id
    return index


def resolve_refs_out(nodes: list[Node], label_index: dict[str, str]) -> None:
    for n in nodes:
        if not n.text_raw:
            continue
        refs = []
        for m in CLAUSE_REF_RE.finditer(n.text_raw):
            label = m.group(1) or m.group(2)
            if not label:
                continue
            refs.append(
                {
                    "raw": m.group(0),
                    "resolved_node_id": label_index.get(label),
                    "type": "internal",
                }
            )
        for m in LEGAL_CITATION_RE.finditer(n.text_raw):
            refs.append({"raw": m.group(0), "resolved_node_id": None, "type": "external_law"})
        if refs:
            n.refs_out = refs


def tag_modality(nodes: list[Node]) -> None:
    for n in nodes:
        text_lower = (n.text_raw or "").lower()
        words = set(re.findall(r"[a-zà-ÿ]+", text_lower))
        flags = n.extraction.setdefault("flags", [])
        if words & NEGATION_TERMS:
            flags.append("negation_present")
        if words & DEONTIC_PROHIBITION:
            flags.append("deontic:dilarang")
        elif words & DEONTIC_OBLIGATION:
            flags.append("deontic:wajib")
        elif words & DEONTIC_PERMISSION:
            flags.append("deontic:dapat")
        if PLACEHOLDER_RE.search(n.text_raw or ""):
            flags.append("template_placeholder")


def harvest_abbreviations(full_text: str) -> list[dict]:
    glossary = []
    seen = set()
    for m in DISINGKAT_RE.finditer(full_text):
        phrase, abbr = m.group(1).strip(), m.group(2).strip()
        key = (phrase.lower(), abbr)
        if key in seen:
            continue
        seen.add(key)
        glossary.append(
            {
                "type": "abbreviation",
                "abbreviation": abbr,
                "expansion": phrase,
                "char_span": [m.start(), m.end()],
            }
        )
    return glossary


def harvest_defined_terms(full_text: str) -> list[dict]:
    terms = []
    seen = set()
    for m in DISEBUT_RE.finditer(full_text):
        phrase, term = m.group(1).strip(), m.group(2).strip()
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(
            {
                "type": "defined_term",
                "term": term,
                "definition_context": phrase[-120:],
                "char_span": [m.start(), m.end()],
            }
        )
    return terms


def harvest_legal_citations(full_text: str) -> list[dict]:
    out = []
    for m in LEGAL_CITATION_RE.finditer(full_text):
        out.append(
            {
                "type": "legal_citation",
                "raw": m.group(0),
                "instrument": m.group(1).upper(),
                "number": m.group(2),
                "year": m.group(3),
                "char_span": [m.start(), m.end()],
            }
        )
    return out


def harvest_rate_equivalents(full_text: str) -> list[dict]:
    """Groups distinct surface forms (1‰, 1/1000, satu per seribu) that encode
    the same rate, so a search for any form finds all provisions using it."""
    forms_by_value: dict[float, set[str]] = {}
    for m in PERCENT_PERMILLE_FRACTION_RE.finditer(full_text):
        raw = m.group(0)
        val = parse_rate(raw)
        if val is None:
            continue
        forms_by_value.setdefault(round(val, 6), set()).add(raw.strip())

    out = []
    for val, forms in forms_by_value.items():
        out.append({"type": "rate", "value": val, "equivalent_forms": sorted(forms)})
    return out


def harvest_placeholders(nodes: list[Node]) -> list[dict]:
    out = []
    for n in nodes:
        for m in PLACEHOLDER_RE.finditer(n.text_raw or ""):
            out.append(
                {
                    "type": "template_placeholder",
                    "raw": m.group(0),
                    "node_id": n.node_id,
                    "char_span": [m.start(), m.end()],
                }
            )
    return out


def extract_document_entities(nodes: list[Node], full_text: str) -> list[dict]:
    entities: list[dict] = []
    entities += harvest_abbreviations(full_text)
    entities += harvest_defined_terms(full_text)
    entities += harvest_legal_citations(full_text)
    entities += harvest_rate_equivalents(full_text)
    entities += harvest_placeholders(nodes)

    for m in NIP_RE.finditer(full_text):
        entities.append({"type": "nip", "raw": m.group(0), "value": m.group(1), "char_span": [m.start(), m.end()]})

    return entities
