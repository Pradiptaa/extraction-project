"""Keyword extraction over an already-built `raw_extraction.json`.

Deliberately a POST-processing step, not a pipeline stage. It consumes the
finished node tree the same way `evaluate.py` and `sample_review.py` do, and
never imports from `pipeline/` — so it works identically on output from the
native pipeline (`pipeline.main`) and the OCR pipeline (`pipeline.ocr_main`),
which emit the same schema.

Two statistical backends are available, selected via `build_body(method=...)`:

  - **rake** (default; Rose et al. 2010): implemented directly rather than via
    `rake-nltk`, which assumes NLTK's English-centric tokenizer and a separate
    corpus download — this project already has curated Indonesian tokenization
    (`_TOKEN_RE`) and a stopword list, so reimplementing the ~30-line scoring
    core avoids an extra heavyweight, English-first dependency. RAKE splits
    text into candidate phrases at stopword/punctuation boundaries, then scores
    each word by degree/frequency (how many distinct words it co-occurs with,
    relative to how often it appears) and sums word scores into a phrase score.
    No external dependency, so it is always available.
  - **yake**: unsupervised, needs no model or training corpus, scores a phrase
    from its position/casing/co-occurrence statistics within THIS document.
    Originally the default; kept available via `method="yake"`. Chosen
    originally over TF-IDF (needs a multi-document corpus to make "rare" mean
    anything — there is one contract, nothing to compare against) and KeyBERT
    (better semantic quality, but pulls in a transformer and embedding weights
    and is not reproducible across library versions).

Neither backend re-derives the core fields statistically — those are already
regex-resolved and ground-truth verified, so they are seeded in as guaranteed
terms and the statistical backend only fills out the remainder from body text.

Worth setting expectations correctly: both backends are single-document and
frequency-based, so neither can distinguish "frequent because it's boilerplate
template language" from "frequent because it matters to this case" — that
needs a multi-document corpus to compare against (see TF-IDF above), which
this project does not have yet with one sample contract.
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


# Node types whose `title` is drawn from a fixed, external outline rather than
# authored for this document. `clause`/`article`/`section` titles are the SSUK/
# SSKK/Pasal table of contents defined by the contract's PROFILE (Perpres
# 16/2018 for this family) — "Jaminan Pelaksanaan", "Cacat Mutu", "Rapat
# Persiapan Pelaksanaan" appear near-verbatim in every contract using this
# profile, so mining them produces terms that identify the template, not the
# document. Confirmed by checking `output/raw_extraction.json` directly: 6 of 8
# generic terms flagged in review were exact clause-title fragments. Body prose
# under these nodes (`text_raw`) is still mined — it is where duration figures,
# named parties, and case-specific clause content actually live.
#
# `heading`/`caption`/`header` are excluded from this rule because their title
# IS the content (a letterhead, a table caption) — nothing external assigned
# it. `subclause`/`list_item` carry no title in this schema (see `tree.py`),
# so the rule has no effect on them either way.
OUTLINE_NODE_TYPES = frozenset({"clause", "article", "section"})


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


_ACRONYM_MAX_LEN = 3


def _normalize_display_case(phrase: str) -> str:
    """Recase a mined phrase for consistent display, word by word.

    YAKE keeps whatever casing the source line used, so raw output mixes
    ALL-CAPS (letterhead lines, table headers — "PENYEDIA", "HARGA KONTRAK"),
    Title Case (ordinary prose), and lowercase in one list. Beyond looking
    inconsistent, this is what let case variants of the same phrase both
    survive dedup — see `_dedupe` below. Recasing up front makes duplicates
    actually collide.

    Only a fully-uppercase, alphabetic word longer than `_ACRONYM_MAX_LEN` is
    touched. Short all-caps tokens are left alone on the assumption they are
    acronyms ("PPK", "RKK") — recasing those to "Ppk" would read worse, not
    better. Seeded core-field values (names, identifiers) never pass through
    this function; they keep the exact casing already resolved from the
    source.
    """
    words = []
    for word in phrase.split(" "):
        if word.isalpha() and word.isupper() and len(word) > _ACRONYM_MAX_LEN:
            word = word[:1] + word[1:].lower()
        words.append(word)
    return " ".join(words)


def _dedupe(terms: list[str]) -> list[str]:
    """Case-insensitive, containment-aware, order-independent.

    Two phrases collapse into one if either's text is contained in the
    other's, compared case-insensitively — checked in BOTH directions. The
    previous version only checked one direction (a new term dropped if it was
    inside an already-kept one), which let a short ALL-CAPS fragment coexist
    with the longer phrase it came from whenever the fragment happened to be
    mined first — e.g. "PENYEDIA" survived alongside "Penyedia Pekerjaan
    Konstruksi" because nothing ever checked whether an EXISTING kept term was
    contained in a NEW, longer one. When two variants do collide, the longer
    phrase is kept, since it is the more specific/informative of the two.
    """
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
    """Split text into RAKE candidate phrases. RAKE's defining rule (Rose et
    al. 2010): a stopword or a phrase-delimiting punctuation mark always ends
    the current candidate — words survive into a phrase together only by
    appearing in an unbroken, stopword-free run. Original word casing is kept
    (for display later); only the stopword lookup below is case-folded."""
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
    """Classic RAKE word/phrase scoring.

    `word_score = degree(word) / frequency(word)`, where `frequency` is how
    often the word appears across all candidates and `degree` is how many
    word-slots it co-occurs with (including itself) — a word that keeps
    company with long phrases and many different neighbours scores higher than
    one that only ever appears alone. A phrase's score is the sum of its
    words' scores. Returns a dict keyed by the phrase's lowercase form (so
    repeated occurrences collapse to their best-scoring instance) mapping to
    (display-cased phrase, score) — RAKE's convention is HIGHER score is
    stronger, the opposite of YAKE's.
    """
    freq: dict[str, int] = {}
    degree: dict[str, int] = {}
    for phrase in phrases:
        co_occurring = len(phrase) - 1
        for word in phrase:
            key = word.lower()
            freq[key] = freq.get(key, 0) + 1
            degree[key] = degree.get(key, 0) + co_occurring
    for key in freq:
        degree[key] += freq[key]  # a word co-occurs with itself once per appearance

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
    """RAKE extraction. Returns (phrase, score) with a HIGHER score meaning a
    stronger keyword — the reverse of `extract_keywords`'s YAKE convention, so
    callers that mix backends must not compare raw scores across them."""
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
    """The `body` field: seeded core-field terms first, then statistically
    extracted keywords, deduped across both. `method` selects the statistical
    backend — "rake" (default) or "yake"."""
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
