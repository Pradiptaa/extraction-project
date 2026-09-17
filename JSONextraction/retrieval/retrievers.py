"""Retrieval strategies behind one interface, so the gate can score them. All
return the same `Hit` shape: dense (embeddings + Chroma), BM25 (lexical, no API
calls, and its IDF discounts the shared boilerplate cosine similarity does not),
and hybrid, fused by Reciprocal Rank Fusion.

RRF rather than a weighted blend: cosine distance and BM25 scores are on
incomparable scales, so blending would need a per-corpus weight. RRF uses only
rank.

Ties are abundant here (much of the corpus is duplicate text) and every ranker
leaves equal-scoring rows in backend order, so compare scores, not orderings.

Scoping (`search(..., scope=...)`) is applied inside each retriever, never to a
finished list — all six specimens are the same standard form, so a top-k taken
before filtering is mostly the wrong document. BM25 and brute force score the
whole corpus anyway, so their scope is an exact pre-sort mask; only dense pushes
it into the index as a Chroma `where`. BM25's IDF stays corpus-wide under a
scope, which keeps scoped and unscoped results comparable.
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
    """One retrieved row. `score` is interpreted per-retriever; see
    `Retriever.score_label`."""

    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""


class Retriever(Protocol):
    name: str
    score_label: str

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        """Best first, at most k. `scope` restricts to rows whose `document_key`
        is in the set; None searches the whole corpus."""


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")

# Function words plus near-universal structural words. Kept short on purpose:
# an aggressive list starts deleting legal terms.
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
    """Sastrawi, wrapped in a cache. Indonesian is affix-heavy, so a lexical
    matcher needs stemming; Sastrawi is slow and the vocabulary repeats
    heavily, so the cache does nearly all the work."""

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


def scope_filter(scope: set[str] | None) -> dict | None:
    """The Chroma `where` clause for a scope, or None for the whole corpus."""
    if not scope:
        return None
    return {"document_key": {"$in": sorted(scope)}}


def tokenizer(name: str):
    """`plain` | `nostop` | `stem`. Which is best is still an open question —
    the earlier numbers predate the gate being tie-stable."""
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
    """Embed the query, ask Chroma for its nearest rows."""

    name = "dense"
    score_label = "dist"

    def __init__(self, collection, embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        # The one retriever where scope changes how the search runs: Chroma
        # applies `where` during traversal. Cross-check against `brute`.
        got = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=scope_filter(scope),
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
    """Exact cosine search over every stored vector — not a production strategy
    but the reference ceiling. A dense score below this is index loss; the gap
    from here to a perfect score is the retriever's real weakness."""

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

        self.document_keys = [str(m.get("document_key") or "") for m in self.metadatas]

        vectors = np.asarray(got["embeddings"], dtype=np.float32)
        self._normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        logger.info("brute-force index: %d vectors held in memory", len(self.ids))

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        np = self._np
        vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        vector /= np.linalg.norm(vector)

        similarities = self._normalised @ vector
        if scope:
            # Out-of-scope rows are pushed below every in-scope one rather than
            # dropped after the sort, so nothing in scope can be missed.
            mask = np.array([key in scope for key in self.document_keys])
            similarities = np.where(mask, similarities, -np.inf)
            k = min(k, int(mask.sum()))
            if k == 0:
                return []
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
    """Lexical BM25 over the same text that was embedded, read from Chroma so
    the two cannot diverge. Pure Python, no API key, built once per instance."""

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
        self.document_keys = [str(m.get("document_key") or "") for m in self.metadatas]

        corpus = [self._tokenize(text or "") for text in self.texts]
        empty = sum(1 for tokens in corpus if not tokens)
        if empty:
            # These score 0 against every query and are lexically unreachable.
            logger.warning(
                "%d of %d documents tokenise to nothing under %r and are lexically unreachable",
                empty, len(corpus), tokenizer_name,
            )
        self._bm25 = BM25Okapi(corpus)
        logger.info("bm25 index: %d documents, tokenizer=%s", len(corpus), tokenizer_name)

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        # Every row is scored anyway, so scope narrows candidates before the
        # sort. IDF and average length stay corpus-wide; see the module docstring.
        scores = self._bm25.get_scores(tokens)
        candidates = range(len(scores))
        if scope:
            candidates = [i for i in candidates if self.document_keys[i] in scope]
        order = sorted(candidates, key=lambda i: -scores[i])[:k]
        # Zero means no query term occurs at all — padding, not a result.
        return [
            Hit(id=self.ids[i], score=float(scores[i]), metadata=self.metadatas[i], text=self.texts[i] or "")
            for i in order
            if scores[i] > 0
        ]


class HybridRetriever:
    """Reciprocal Rank Fusion over a dense and a lexical ranking. Each is asked
    for `pool` candidates and the fused list cut to k; the pool must exceed k or
    fusion has nothing to work with.

    `pool` is not a quality knob — widening it changes which rows Chroma
    explores and can return a worse top-k. Index quality is `config.INDEX_METADATA`.
    """

    name = "hybrid"
    score_label = "rrf"

    def __init__(self, dense: Retriever, lexical: Retriever, pool: int = 50, rrf_k: int = RRF_K) -> None:
        self.dense = dense
        self.lexical = lexical
        self.pool = pool
        self.rrf_k = rrf_k

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        pool = max(self.pool, k)
        # Scope goes to both sides, never the fused list, or most of the pool
        # is spent on other contracts holding the same standard-form clause.
        rankings = [
            self.dense.search(query, pool, scope),
            self.lexical.search(query, pool, scope),
        ]

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
