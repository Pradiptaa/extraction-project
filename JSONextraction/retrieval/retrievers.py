from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

RRF_K = 60 

@dataclass
class Hit:
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

_WORD = re.compile(r"[0-9]+(?:\.[0-9]+)+|[\w]+", re.UNICODE)

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
    if not scope:
        return None
    return {"document_key": {"$in": sorted(scope)}}


def tokenizer(name: str):
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

    name = "dense"
    score_label = "dist"

    def __init__(self, collection, embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
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
            mask = np.array([key in scope for key in self.document_keys])
            similarities = np.where(mask, similarities, -np.inf)
            k = min(k, int(mask.sum()))
            if k == 0:
                return []
        order = sorted(range(len(self.ids)), key=lambda i: (-similarities[i], self.ids[i]))[:k]
        return [
            Hit(
                id=self.ids[i],
                score=float(1.0 - similarities[i]), 
                metadata=self.metadatas[i],
                text=self.texts[i] or "",
            )
            for i in order
        ]


class Bm25Retriever:

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
        scores = self._bm25.get_scores(tokens)
        candidates = range(len(scores))
        if scope:
            candidates = [i for i in candidates if self.document_keys[i] in scope]
        order = sorted(candidates, key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [
            Hit(id=self.ids[i], score=float(scores[i]), metadata=self.metadatas[i], text=self.texts[i] or "")
            for i in order
            if scores[i] > 0
        ]


class HybridRetriever:

    name = "hybrid"
    score_label = "rrf"

    def __init__(self, dense: Retriever, lexical: Retriever, pool: int = 50, rrf_k: int = RRF_K) -> None:
        self.dense = dense
        self.lexical = lexical
        self.pool = pool
        self.rrf_k = rrf_k

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        pool = max(self.pool, k)
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

        ordered = sorted(fused, key=lambda row_id: (-fused[row_id], row_id))[:k]
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
        return HybridRetriever(
            BruteForceRetriever(collection, embedder),
            Bm25Retriever(collection, tokenizer_name),
            pool=pool,
        )
    raise ValueError(
        f"unknown retriever {name!r} (expected dense, brute, bm25, hybrid or hybrid-brute)"
    )
