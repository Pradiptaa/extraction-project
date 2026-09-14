"""Retrieval strategies behind one interface, so the gate can score them.

`retrieval_evaluate.py` used to call `collection.query` inline, which meant
there was no seam to compare configurations at — and comparison is the whole
point of adding a second strategy. Everything here returns the same `Hit` shape,
so the harness neither knows nor cares which one produced a result.

Three strategies:

- `DenseRetriever`   — embeddings + Chroma, what the gate has always measured.
- `Bm25Retriever`    — lexical scoring, no API calls at all. Worth having on a
  corpus where 25-39% of each document is shared boilerplate: IDF discounts a
  phrase that appears in every contract, which cosine similarity does not.
- `HybridRetriever`  — both, fused by Reciprocal Rank Fusion.

RRF rather than a weighted blend of scores, deliberately: cosine distance and
BM25 scores are on incomparable scales with no principled conversion between
them, so blending needs a weight tuned per corpus. RRF uses only rank, so it has
one constant and nothing to fit — which also keeps it clear of tuning
parameters until the gate goes green.

A note on ties, which this corpus has in abundance (60% of rows are duplicate
text): every ranker here leaves equal-scoring rows in whatever order the backend
returned them. That is fine for scoring because the gate accepts text-equivalent
rows (see `retrieval_evaluate.accepted_rows`), but it means a raw result list is
not stable enough to diff between runs. Compare scores, not orderings.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard constant from the original RRF paper; not tuned here.


@dataclass
class Hit:
    """One retrieved row. `score` is interpreted per-retriever — see
    `Retriever.score_label` — because a distance (lower is better) and an RRF
    weight (higher is better) cannot share a scale."""

    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""


class Retriever(Protocol):
    name: str
    score_label: str

    def search(self, query: str, k: int) -> list[Hit]:
        """Best first, at most k."""


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")

# High-frequency Indonesian function words, plus the structural words that
# appear in nearly every node of every specimen ("pasal", "ayat", "huruf").
# Kept short on purpose: an aggressive list starts deleting legal terms.
STOPWORDS = frozenset("""
yang dan di ke dari untuk dengan pada dalam atau adalah ini itu akan tidak
dapat oleh sebagai telah harus sudah juga bila jika maka agar serta antara
setelah sebelum atas bawah para nya yaitu yakni tersebut terhadap secara bahwa
hal lain nomor tanggal huruf ayat angka
""".split())


def tokenize_plain(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def tokenize_no_stopwords(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class _Stemmer:
    """Sastrawi, wrapped in a cache.

    Indonesian is affix-heavy — `pekerjaan`, `mengerjakan` and `dikerjakan`
    share a root that a lexical matcher otherwise treats as three unrelated
    tokens. Sastrawi is slow enough that stemming 4021 documents uncached is
    noticeable, and the token vocabulary repeats heavily, so the cache does
    nearly all the work.
    """

    def __init__(self) -> None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        self._stemmer = StemmerFactory().create_stemmer()
        self._cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        out = []
        for token in _WORD.findall(text.lower()):
            if token in STOPWORDS:
                continue
            stemmed = self._cache.get(token)
            if stemmed is None:
                stemmed = self._cache[token] = self._stemmer.stem(token)
            if stemmed:
                out.append(stemmed)
        return out


def tokenizer(name: str):
    """`plain` | `nostop` | `stem`.

    Which one is best is an open question: the exploration numbers that seemed
    to answer it were measured before the gate was tie-stable, so they are
    void. Re-derive the choice against the current gate.
    """
    if name == "plain":
        return tokenize_plain
    if name == "nostop":
        return tokenize_no_stopwords
    if name == "stem":
        return _Stemmer()
    raise ValueError(f"unknown tokenizer {name!r} (expected plain, nostop or stem)")


# --------------------------------------------------------------------------
# retrievers
# --------------------------------------------------------------------------


class DenseRetriever:
    """Embed the query, ask Chroma for its nearest rows.

    Behaviour-identical to the inline query this replaced, so the gate's
    recorded baseline still means what it meant.
    """

    name = "dense"
    score_label = "dist"

    def __init__(self, collection, embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, k: int) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        got = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["metadatas", "distances", "documents"],
        )
        ids = got["ids"][0]
        documents = (got.get("documents") or [[]])[0] or [""] * len(ids)
        return [
            Hit(id=row_id, score=distance, metadata=dict(metadata or {}), text=text or "")
            for row_id, distance, metadata, text in zip(
                ids, got["distances"][0], got["metadatas"][0], documents
            )
        ]


class BruteForceRetriever:
    """Exact cosine search over every stored vector. No index, no approximation.

    Not a production strategy — it scores the whole collection per query. It is
    the **reference ceiling**: the answer to "is the index hiding something, or
    is the ranking genuinely weak", which must be settled before any
    claim about ranking. A dense score below this one is index loss; the gap
    between this and a perfect score is the retriever's real weakness.

    Keeping it here rather than in a scratch script means that check is a flag
    away (`--retriever brute`) instead of something to re-derive each time.
    """

    name = "brute"
    score_label = "dist"

    def __init__(self, collection, embedder) -> None:
        import numpy as np

        self._np = np
        self.embedder = embedder

        got = collection.get(include=["documents", "metadatas", "embeddings"])
        self.ids = got["ids"]
        self.texts = got.get("documents") or [""] * len(self.ids)
        self.metadatas = [dict(m or {}) for m in (got.get("metadatas") or [{}] * len(self.ids))]

        vectors = np.asarray(got["embeddings"], dtype=np.float32)
        self._normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        logger.info("brute-force index: %d vectors held in memory", len(self.ids))

    def search(self, query: str, k: int) -> list[Hit]:
        np = self._np
        vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        vector /= np.linalg.norm(vector)

        similarities = self._normalised @ vector
        order = np.argsort(-similarities)[:k]
        return [
            Hit(
                id=self.ids[i],
                score=float(1.0 - similarities[i]),  # cosine distance, to match DenseRetriever
                metadata=self.metadatas[i],
                text=self.texts[i] or "",
            )
            for i in order
        ]


class Bm25Retriever:
    """Lexical BM25 over the same text that was embedded.

    The corpus is read from Chroma rather than from the embedding views on
    disk, which was verified to be safe: all 4021 documents round-trip
    byte-identically. That matters — indexing text that
    differs from what was embedded would make any hybrid comparison meaningless.

    The index is built once per instance. It is pure Python and needs no API
    key, so this retriever is free to run.
    """

    name = "bm25"
    score_label = "bm25"

    def __init__(self, collection, tokenizer_name: str = "plain") -> None:
        from rank_bm25 import BM25Okapi

        self.tokenizer_name = tokenizer_name
        self._tokenize = tokenizer(tokenizer_name)

        got = collection.get(include=["documents", "metadatas"])
        self.ids = got["ids"]
        self.texts = got.get("documents") or [""] * len(self.ids)
        self.metadatas = [dict(m or {}) for m in (got.get("metadatas") or [{}] * len(self.ids))]

        corpus = [self._tokenize(text or "") for text in self.texts]
        empty = sum(1 for tokens in corpus if not tokens)
        if empty:
            # BM25 scores these 0 against every query, so they can never be
            # retrieved lexically. Worth knowing rather than silently carrying.
            logger.warning(
                "%d of %d documents tokenise to nothing under %r and are lexically unreachable",
                empty, len(corpus), tokenizer_name,
            )
        self._bm25 = BM25Okapi(corpus)
        logger.info("bm25 index: %d documents, tokenizer=%s", len(corpus), tokenizer_name)

    def search(self, query: str, k: int) -> list[Hit]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        # A zero score means no query term occurs in the document; returning
        # those would pad the list with rows BM25 has no opinion about.
        return [
            Hit(id=self.ids[i], score=float(scores[i]), metadata=self.metadatas[i], text=self.texts[i] or "")
            for i in order
            if scores[i] > 0
        ]


class HybridRetriever:
    """Reciprocal Rank Fusion over a dense and a lexical ranking.

    Each retriever is asked for `pool` candidates, and the fused list is cut to
    k. The pool has to exceed k or fusion has nothing to work with: a row that
    is rank 8 dense and rank 2 lexical is exactly the kind of hit hybrid search
    exists to surface, and a pool of k would never see it.

    `pool` is NOT a quality knob to turn up. Asking Chroma for more results
    changes which rows it explores and can return a worse top-k — measured at
    9/16 versus 10/16 for a plain widening. Pool size is a
    fusion input; index quality is set by `config.INDEX_METADATA`.
    """

    name = "hybrid"
    score_label = "rrf"

    def __init__(self, dense: Retriever, lexical: Retriever, pool: int = 50, rrf_k: int = RRF_K) -> None:
        self.dense = dense
        self.lexical = lexical
        self.pool = pool
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        pool = max(self.pool, k)
        rankings = [self.dense.search(query, pool), self.lexical.search(query, pool)]

        hits: dict[str, Hit] = {}
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                hits.setdefault(hit.id, hit)
                fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused, key=lambda row_id: -fused[row_id])[:k]
        return [Hit(id=i, score=fused[i], metadata=hits[i].metadata, text=hits[i].text) for i in ordered]


def build_retriever(name: str, collection, embedder, pool: int = 50, tokenizer_name: str = "plain") -> Retriever:
    if name == "dense":
        return DenseRetriever(collection, embedder)
    if name == "brute":
        return BruteForceRetriever(collection, embedder)
    if name == "bm25":
        return Bm25Retriever(collection, tokenizer_name)
    if name == "hybrid":
        return HybridRetriever(
            DenseRetriever(collection, embedder),
            Bm25Retriever(collection, tokenizer_name),
            pool=pool,
        )
    if name == "hybrid-brute":
        # The ceiling for hybrid: fusion over an exact dense ranking, so a gap
        # against plain `hybrid` is index loss rather than a fusion problem.
        return HybridRetriever(
            BruteForceRetriever(collection, embedder),
            Bm25Retriever(collection, tokenizer_name),
            pool=pool,
        )
    raise ValueError(
        f"unknown retriever {name!r} (expected dense, brute, bm25, hybrid or hybrid-brute)"
    )
