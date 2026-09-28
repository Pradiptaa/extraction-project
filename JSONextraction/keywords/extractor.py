from __future__ import annotations

import re
import unicodedata
from pathlib import Path

DEFAULT_STOPWORDS_PATH = Path(__file__).with_name("stopwords_id.txt")

RESCUED_TERMS = frozenset({"pihak", "waktu", "bagian"})

DOMAIN_STOPWORDS = frozenset({
    "tersebut", "dimaksud", "berikut", "sebagaimana", "selanjutnya", "disebut",
    "yaitu", "antara", "lain", "atau", "dan/atau", "serta", "guna",
    "terhadap", "mengenai", "melalui", "berdasarkan", "sesuai", "meliputi",
    "adapun", "bahwa", "maka", "apabila", "jika", "kecuali", "sedangkan",
    "dst", "dll", "tsb", "ybs",
})

STRUCTURAL_TERMS = frozenset({
    "pasal", "ayat", "bab", "huruf", "angka", "lampiran", "butir",
    "halaman", "nomor", "no", "hal",
})

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_MIN_TERM_LEN = 3
SIMILARITY_LIMIT = 0.5


def load_stopwords(path: Path | None = None) -> set[str]:
    source = path or DEFAULT_STOPWORDS_PATH
    words: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return (words - RESCUED_TERMS) | DOMAIN_STOPWORDS | STRUCTURAL_TERMS


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def _is_useful(term: str, stopwords: set[str]) -> bool:
    if len(term) < _MIN_TERM_LEN:
        return False
    tokens = _TOKEN_RE.findall(term.lower())
    if not tokens:
        return False
    if len(set(tokens)) != len(tokens):
        return False
    return any(t not in stopwords and len(t) >= _MIN_TERM_LEN for t in tokens)


def _content_tokens(term: str, stopwords: set[str]) -> frozenset[str]:
    return frozenset(
        t for t in _TOKEN_RE.findall(term.lower())
        if t not in stopwords and len(t) >= _MIN_TERM_LEN
    )


def _too_similar(tokens: frozenset[str], kept: list[frozenset[str]], threshold: float) -> bool:
    if not tokens:
        return True
    for other in kept:
        union = tokens | other
        if union and len(tokens & other) / len(union) >= threshold:
            return True
    return False


OUTLINE_NODE_TYPES = frozenset({"clause", "article", "section"})


def document_segments(document: dict) -> list[str]:
    segments: list[str] = []
    for node in document.get("structure") or []:
        fields = ("text_raw",) if node.get("node_type") in OUTLINE_NODE_TYPES else ("title", "text_raw")
        for field in fields:
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
    parts = []
    for segment in document_segments(document):
        parts.append(segment if segment.endswith((".", ";", ":", "?", "!")) else segment + ".")
    return "\n".join(parts)


def seed_terms(document: dict) -> list[str]:
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

    for number in _value("key_numbers") or []:
        if isinstance(number, dict) and number.get("type") == "reference_number":
            value = number.get("value") or number.get("raw")
            if isinstance(value, str) and value.strip():
                seeds.append(normalize(value))

    return _dedupe(seeds)


_ACRONYM_MAX_LEN = 3


def _normalize_display_case(phrase: str) -> str:
    words = []
    for word in phrase.split(" "):
        if word.isalpha() and word.isupper() and len(word) > _ACRONYM_MAX_LEN:
            word = word[:1] + word[1:].lower()
        words.append(word)
    return " ".join(words)


def _dedupe(terms: list[str]) -> list[str]:
    kept: list[str] = []
    for term in terms:
        low = term.lower()
        replaced = False
        drop = False
        for i, existing in enumerate(kept):
            existing_low = existing.lower()
            if low == existing_low or low in existing_low:
                drop = True
                break
            if existing_low in low:
                kept[i] = term
                replaced = True
                break
        if not drop and not replaced:
            kept.append(term)
    return kept


def extract_keywords(
    text: str,
    stopwords: set[str],
    top_n: int = 40,
    ngram_max: int = 3,
) -> list[tuple[str, float]]:
    try:
        import yake
    except ImportError as exc: 
        raise SystemExit(
            "yake is required for keyword extraction: pip install -r requirements.txt"
        ) from exc

    if not text.strip():
        return []

    extractor = yake.KeywordExtractor(
        lan="id",
        n=ngram_max,
        top=top_n * 10, 
        dedupLim=0.9,
        stopwords=stopwords,
    )
    scored = [
        (_normalize_display_case(normalize(phrase)), score)
        for phrase, score in extractor.extract_keywords(text)
    ]
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


_RAKE_SPLIT_RE = re.compile(r"[,.;:!?()\[\]{}\"'–—|/]+")
_RAKE_MAX_PHRASE_WORDS = 4


def _rake_candidate_phrases(text: str, stopwords: set[str]) -> list[list[str]]:
    phrases: list[list[str]] = []
    for chunk in _RAKE_SPLIT_RE.split(text):
        current: list[str] = []
        for word in _TOKEN_RE.findall(chunk):
            if word.lower() in stopwords or len(word) < _MIN_TERM_LEN:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(word)
        if current:
            phrases.append(current)
    return [p for p in phrases if len(p) <= _RAKE_MAX_PHRASE_WORDS]


def _rake_score_phrases(phrases: list[list[str]]) -> dict[str, tuple[str, float]]:
    freq: dict[str, int] = {}
    degree: dict[str, int] = {}
    for phrase in phrases:
        co_occurring = len(phrase) - 1
        for word in phrase:
            key = word.lower()
            freq[key] = freq.get(key, 0) + 1
            degree[key] = degree.get(key, 0) + co_occurring
    for key in freq:
        degree[key] += freq[key] 

    word_score = {key: degree[key] / freq[key] for key in freq}

    best: dict[str, tuple[str, float]] = {}
    for phrase in phrases:
        key = " ".join(w.lower() for w in phrase)
        score = sum(word_score[w.lower()] for w in phrase)
        if key not in best or score > best[key][1]:
            best[key] = (" ".join(phrase), score)
    return best


def extract_keywords_rake(
    text: str,
    stopwords: set[str],
    top_n: int = 40,
) -> list[tuple[str, float]]:
    if not text.strip():
        return []

    phrases = _rake_candidate_phrases(text, stopwords)
    if not phrases:
        return []

    scored = sorted(_rake_score_phrases(phrases).values(), key=lambda pair: -pair[1])
    cased = [(_normalize_display_case(normalize(phrase)), score) for phrase, score in scored]
    useful = [(phrase, score) for phrase, score in cased if _is_useful(phrase, stopwords)]

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


def build_body(
    document: dict,
    top_n: int = 40,
    stopwords: set[str] | None = None,
    method: str = "rake",
) -> list[str]:
    words = stopwords if stopwords is not None else load_stopwords()
    seeds = seed_terms(document)
    text = document_text(document)

    if method == "yake":
        mined = [phrase for phrase, _ in extract_keywords(text, words, top_n=top_n)]
    elif method == "rake":
        mined = [phrase for phrase, _ in extract_keywords_rake(text, words, top_n=top_n)]
    else:
        raise ValueError(f"unknown keyword extraction method: {method!r} (expected 'yake' or 'rake')")

    return _dedupe(seeds + mined)
