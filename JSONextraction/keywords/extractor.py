"""Keyword extraction over an already-built `raw_extraction.json`.

Deliberately a POST-processing step, not a pipeline stage. It consumes the
finished node tree the same way `evaluate.py` and `sample_review.py` do, and
never imports from `pipeline/` — so it works identically on output from the
native pipeline (`pipeline.main`) and the OCR pipeline (`pipeline.ocr_main`),
which emit the same schema.

Why YAKE rather than the alternatives:
  - TF-IDF needs a multi-document corpus to make "rare" mean anything. There is
    one contract today, so it has nothing to measure rarity against.
  - KeyBERT scores best on semantic benchmarks but pulls in a transformer and
    embedding weights, and its output is not reproducible across versions.
  - YAKE is unsupervised, single-document, needs no model or training corpus,
    and is deterministic given the same input — which is the same property the
    rest of this project insists on (no LLM, no black box; see the pipeline's
    scope boundary).

The core fields are NOT re-derived statistically. They are already regex-
resolved and ground-truth verified, so they are seeded in as guaranteed terms
and YAKE only fills out the remainder from the body text.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

DEFAULT_STOPWORDS_PATH = Path(__file__).with_name("stopwords_id.txt")

# Present in the upstream generic list, but load-bearing in a contract: "pihak"
# (party) is the single most important noun in the document, "waktu" carries
# every duration clause, and "bagian" names structural divisions. A generic
# Indonesian stopword list is built for prose, not for legal text.
RESCUED_TERMS = frozenset({"pihak", "waktu", "bagian"})

# Contract boilerplate that IS noise here but is too domain-specific to appear
# in a general list. These are template scaffolding, not subject matter.
DOMAIN_STOPWORDS = frozenset({
    "tersebut", "dimaksud", "berikut", "sebagaimana", "selanjutnya", "disebut",
    "yaitu", "antara", "lain", "atau", "dan/atau", "serta", "guna",
    "terhadap", "mengenai", "melalui", "berdasarkan", "sesuai", "meliputi",
    "adapun", "bahwa", "maka", "apabila", "jika", "kecuali", "sedangkan",
    "dst", "dll", "tsb", "ybs",
})

# Structural vocabulary kept OUT of the keyword body: these describe the
# document's shape, which `structure[]` already records far more precisely than
# a keyword ever could. Including them buries the subject matter under
# scaffolding — every contract has "pasal", only this one has "Natai Sedawak".
STRUCTURAL_TERMS = frozenset({
    "pasal", "ayat", "bab", "huruf", "angka", "lampiran", "butir",
    "halaman", "nomor", "no", "hal",
})

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_MIN_TERM_LEN = 3
# Jaccard overlap of content tokens at or above which two phrases are treated as
# the same keyword. 0.5 means "half their meaningful words coincide".
SIMILARITY_LIMIT = 0.5


def load_stopwords(path: Path | None = None) -> set[str]:
    """Upstream list, minus the terms this domain needs back, plus the ones it
    additionally considers noise."""
    source = path or DEFAULT_STOPWORDS_PATH
    words: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return (words - RESCUED_TERMS) | DOMAIN_STOPWORDS | STRUCTURAL_TERMS


def normalize(text: str) -> str:
    """Fold to a comparable surface form without destroying it. NFKC repairs
    OCR's compatibility characters; case and accents are left intact so the
    stored keyword stays human-readable."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def _is_useful(term: str, stopwords: set[str]) -> bool:
    if len(term) < _MIN_TERM_LEN:
        return False
    tokens = _TOKEN_RE.findall(term.lower())
    if not tokens:
        return False
    # A phrase repeating a token ("Kontrak Kontrak", "Pekerjaan pekerjaan") is
    # an n-gram window that crossed a boundary, not a real phrase. Indonesian
    # does reduplicate for plurals, but that is written with a hyphen
    # ("syarat-syarat") and survives as a single token here.
    if len(set(tokens)) != len(tokens):
        return False
    # A phrase survives only if it carries at least one non-stopword token;
    # a phrase made entirely of stopwords is scaffolding regardless of score.
    return any(t not in stopwords and len(t) >= _MIN_TERM_LEN for t in tokens)


def _content_tokens(term: str, stopwords: set[str]) -> frozenset[str]:
    return frozenset(
        t for t in _TOKEN_RE.findall(term.lower())
        if t not in stopwords and len(t) >= _MIN_TERM_LEN
    )


def _too_similar(tokens: frozenset[str], kept: list[frozenset[str]], threshold: float) -> bool:
    """Jaccard overlap against everything already kept. Substring dedupe alone
    leaves near-duplicates that share every meaningful word in a different order
    ("Pelaksanaan Kontrak" / "Kontrak Pelaksanaan Pekerjaan"), which is what
    made the first 40 results collapse onto four repeated nouns."""
    if not tokens:
        return True
    for other in kept:
        union = tokens | other
        if union and len(tokens & other) / len(union) >= threshold:
            return True
    return False


def document_segments(document: dict) -> list[str]:
    """Coherent text units to mine, as separate segments.

    Built from the node tree rather than `pages[].raw_text`, because a node is a
    semantic unit (a clause body, a heading) while a page is an arbitrary
    rectangle. Mining page text lets n-grams span a line break and produces
    phrases that occur nowhere in the document — "melaksanakan Kontrak Kontrak"
    and "Pelaksanaan Pekerjaan pekerjaan" were both artefacts of that.

    Table cell text lives in `tables[]` rather than in `structure[]`, so it is
    added separately, one segment per row.
    """
    segments: list[str] = []
    for node in document.get("structure") or []:
        for field in ("title", "text_raw"):
            value = normalize(node.get(field) or "")
            if value:
                segments.append(value)
    for table in document.get("tables") or []:
        for row in table.get("rows") or []:
            cells = [normalize(c) for c in (row.get("cells") or []) if normalize(c)]
            if cells:
                segments.append(" ".join(cells))
    return segments


def document_text(document: dict) -> str:
    """Segments joined so that each terminates a sentence, which is what stops
    YAKE's n-gram window from running across a boundary into the next node."""
    parts = []
    for segment in document_segments(document):
        parts.append(segment if segment.endswith((".", ";", ":", "?", "!")) else segment + ".")
    return "\n".join(parts)


def seed_terms(document: dict) -> list[str]:
    """Guaranteed keywords taken from the already-resolved core fields. These
    are regex-resolved and ground-truth verified, so they are never subject to
    statistical scoring — they are simply always present."""
    core = document.get("core") or {}
    seeds: list[str] = []

    def _value(field: str):
        entry = core.get(field) or {}
        return entry.get("value") if isinstance(entry, dict) else None

    for field in ("contract_name", "contract_number", "document_type"):
        value = _value(field)
        if isinstance(value, str) and value.strip():
            seeds.append(normalize(value))

    for party in _value("parties") or []:
        if not isinstance(party, dict):
            continue
        org = (party.get("organization") or {}).get("value")
        if isinstance(org, str) and org.strip():
            seeds.append(normalize(org))
        rep = (party.get("representative") or {}).get("name")
        if isinstance(rep, str) and rep.strip():
            seeds.append(normalize(rep))

    # Identifier-shaped numbers are high-value search keys; magnitudes and
    # durations are not keywords and stay in `key_numbers`.
    for number in _value("key_numbers") or []:
        if isinstance(number, dict) and number.get("type") == "reference_number":
            value = number.get("value") or number.get("raw")
            if isinstance(value, str) and value.strip():
                seeds.append(normalize(value))

    return _dedupe(seeds)


def _dedupe(terms: list[str]) -> list[str]:
    """Case-insensitive, order-preserving. Also drops any term wholly contained
    in one already kept, which is what collapses YAKE's overlapping n-grams
    ("Peningkatan Jalan" inside "Peningkatan Jalan Mekar Desa Natai Sedawak")."""
    kept: list[str] = []
    lowered: list[str] = []
    for term in terms:
        low = term.lower()
        if any(low == seen or low in seen for seen in lowered):
            continue
        kept.append(term)
        lowered.append(low)
    return kept


def extract_keywords(
    text: str,
    stopwords: set[str],
    top_n: int = 40,
    ngram_max: int = 3,
) -> list[tuple[str, float]]:
    """Statistical keyphrase extraction. Returns (phrase, score) with a LOWER
    score meaning a stronger keyword, which is YAKE's own convention."""
    try:
        import yake
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "yake is required for keyword extraction: pip install -r requirements.txt"
        ) from exc

    if not text.strip():
        return []

    extractor = yake.KeywordExtractor(
        lan="id",
        n=ngram_max,
        # Over-fetch heavily: the similarity filter below discards most of a
        # contract's raw candidates, which cluster hard on a few nouns.
        top=top_n * 10,
        dedupLim=0.9,
        stopwords=stopwords,
    )
    scored = [(normalize(phrase), score) for phrase, score in extractor.extract_keywords(text)]
    useful = [(phrase, score) for phrase, score in scored if _is_useful(phrase, stopwords)]
    useful.sort(key=lambda pair: pair[1])

    kept: list[tuple[str, float]] = []
    kept_tokens: list[frozenset[str]] = []
    for phrase, score in useful:
        tokens = _content_tokens(phrase, stopwords)
        if _too_similar(tokens, kept_tokens, SIMILARITY_LIMIT):
            continue
        kept.append((phrase, score))
        kept_tokens.append(tokens)
        if len(kept) >= top_n:
            break
    return kept


def build_body(document: dict, top_n: int = 40, stopwords: set[str] | None = None) -> list[str]:
    """The `body` field: seeded core-field terms first, then statistically
    extracted keywords, deduped across both."""
    words = stopwords if stopwords is not None else load_stopwords()
    seeds = seed_terms(document)
    mined = [phrase for phrase, _ in extract_keywords(document_text(document), words, top_n=top_n)]
    return _dedupe(seeds + mined)
