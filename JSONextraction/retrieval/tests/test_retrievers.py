"""Unit tests for retrieval.retrievers.

No API key: the dense side uses a fake embedder, and BM25 needs none. The fusion
maths and the tokenisers are testable without knowing what the model thinks
today, which is the same rule the rest of this suite follows.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import chromadb

from retrieval.config import INDEX_METADATA
from retrieval.retrievers import (
    Bm25Retriever,
    BruteForceRetriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    build_retriever,
    tokenize_no_stopwords,
    tokenize_plain,
    tokenizer,
)


class StaticRetriever:
    """Returns a fixed ranking, so fusion can be checked by hand."""

    name = "static"
    score_label = "static"

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(id=i, score=1.0, metadata={"label": i}, text=f"text {i}") for i in self.ids[:k]]


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [list(self.vector) for _ in texts]


ROWS = [
    ("r_denda", [1.0, 0.0, 0.0], "Pembayaran denda keterlambatan penyelesaian pekerjaan"),
    ("r_kahar", [0.0, 1.0, 0.0], "Keadaan kahar force majeure"),
    ("r_hki", [0.0, 0.0, 1.0], "Pelanggaran hak kekayaan intelektual oleh penyedia"),
    ("r_dup", [0.9, 0.1, 0.0], "Pembayaran denda keterlambatan penyelesaian pekerjaan"),
]


class TokenizerTests(unittest.TestCase):
    def test_plain_lowercases_and_splits_on_non_word(self) -> None:
        self.assertEqual(tokenize_plain("Pasal 55.2: Asuransi!"), ["pasal", "55", "2", "asuransi"])

    def test_stopwords_are_removed_but_legal_terms_survive(self) -> None:
        tokens = tokenize_no_stopwords("denda yang dibayar oleh penyedia dalam kontrak ini")
        self.assertIn("denda", tokens)
        self.assertIn("penyedia", tokens)
        self.assertIn("kontrak", tokens)
        for stopword in ("yang", "oleh", "dalam", "ini"):
            self.assertNotIn(stopword, tokens)

    def test_stemmer_collapses_indonesian_affixes(self) -> None:
        """The reason stemming is worth testing at all: Indonesian is
        affix-heavy, so a lexical matcher otherwise treats three forms of one
        root as unrelated tokens."""
        stem = tokenizer("stem")
        self.assertEqual(stem("pekerjaan"), stem("mengerjakan"))

    def test_unknown_tokenizer_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tokenizer("porter")


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        client = chromadb.PersistentClient(path=str(self.tmp / "chroma"))
        self.collection = client.get_or_create_collection("test_rows", metadata=INDEX_METADATA)
        self.collection.add(
            ids=[r[0] for r in ROWS],
            embeddings=[r[1] for r in ROWS],
            metadatas=[{"label": r[0], "sub_document": "general_terms"} for r in ROWS],
            documents=[r[2] for r in ROWS],
        )

    def test_dense_returns_hits_with_text_and_metadata(self) -> None:
        embedder = FakeEmbedder([0.0, 0.0, 1.0])
        hits = DenseRetriever(self.collection, embedder).search("hak kekayaan intelektual", 2)

        self.assertEqual(hits[0].id, "r_hki")
        self.assertEqual(hits[0].text, ROWS[2][2])
        self.assertEqual(hits[0].metadata["label"], "r_hki")
        self.assertEqual(embedder.calls, ["hak kekayaan intelektual"])

    def test_dense_respects_k(self) -> None:
        hits = DenseRetriever(self.collection, FakeEmbedder([1.0, 0.0, 0.0])).search("denda", 2)
        self.assertEqual(len(hits), 2)

    def test_bm25_matches_on_words_without_any_embedding_call(self) -> None:
        hits = Bm25Retriever(self.collection).search("kekayaan intelektual", 3)
        self.assertEqual(hits[0].id, "r_hki")

    def test_bm25_returns_nothing_when_no_term_occurs(self) -> None:
        """Padding the list with zero-scoring rows would hand fusion candidates
        BM25 has no opinion about."""
        self.assertEqual(Bm25Retriever(self.collection).search("zzzz qqqq", 5), [])

    def test_bm25_returns_nothing_for_an_all_stopword_query(self) -> None:
        self.assertEqual(Bm25Retriever(self.collection, "nostop").search("yang dan di ke", 5), [])

    def test_hybrid_promotes_rows_both_sides_agree_on(self) -> None:
        """RRF's whole point: agreement across two rankings outweighs a strong
        showing in one. Dense ranks a,b,c; lexical ranks c,b,d. Both b and c
        appear on both lists and must finish above a (dense-only, rank 1) and d
        (lexical-only) — a row nobody corroborates does not win on one opinion.

        Scores never meet, so the incomparable scales of cosine distance and
        BM25 never have to be reconciled.
        """
        hybrid = HybridRetriever(StaticRetriever(["a", "b", "c"]), StaticRetriever(["c", "b", "d"]), pool=3)
        ordered = [hit.id for hit in hybrid.search("q", 4)]

        self.assertEqual(set(ordered[:2]), {"b", "c"}, "rows on both lists must come first")
        self.assertEqual(set(ordered), {"a", "b", "c", "d"})
        # a is rank 1 dense but absent from lexical, so it still loses to both.
        self.assertGreater(ordered.index("a"), ordered.index("b"))

    def test_hybrid_beats_a_single_list_on_position(self) -> None:
        """Within one list, better rank still wins: fusion reorders, it does not
        discard ranking information."""
        hybrid = HybridRetriever(StaticRetriever(["a", "b", "c"]), StaticRetriever([]), pool=3)
        self.assertEqual([h.id for h in hybrid.search("q", 3)], ["a", "b", "c"])

    def test_hybrid_degrades_to_dense_when_lexical_finds_nothing(self) -> None:
        dense = StaticRetriever(["a", "b", "c"])
        hybrid = HybridRetriever(dense, StaticRetriever([]), pool=3)
        self.assertEqual([h.id for h in hybrid.search("q", 3)], ["a", "b", "c"])

    def test_hybrid_pool_is_never_smaller_than_k(self) -> None:
        """A pool below k would starve fusion of the candidates it exists to
        reorder."""
        hybrid = HybridRetriever(StaticRetriever(list("abcdefgh")), StaticRetriever(list("hgfedcba")), pool=2)
        self.assertEqual(len(hybrid.search("q", 6)), 6)

    def test_hybrid_carries_text_and_metadata_through_fusion(self) -> None:
        """The harness scores on metadata and the chat layer quotes the text, so
        fusion must not reduce a hit to a bare id."""
        hybrid = HybridRetriever(StaticRetriever(["a"]), StaticRetriever(["a"]), pool=2)
        hit = hybrid.search("q", 1)[0]
        self.assertEqual(hit.text, "text a")
        self.assertEqual(hit.metadata, {"label": "a"})

    def test_brute_force_finds_the_exact_nearest_row(self) -> None:
        """The reference ceiling has to be exact, or it cannot diagnose the
        index. Its distances must also be comparable to DenseRetriever's, since
        the whole point is comparing the two."""
        embedder = FakeEmbedder([0.0, 1.0, 0.0])
        brute = BruteForceRetriever(self.collection, embedder).search("kahar", 1)
        dense = DenseRetriever(self.collection, FakeEmbedder([0.0, 1.0, 0.0])).search("kahar", 1)

        self.assertEqual(brute[0].id, "r_kahar")
        self.assertEqual(brute[0].id, dense[0].id)
        self.assertAlmostEqual(brute[0].score, dense[0].score, places=5)

    def test_brute_force_ranks_every_row_not_just_a_pool(self) -> None:
        hits = BruteForceRetriever(self.collection, FakeEmbedder([1.0, 0.0, 0.0])).search("denda", 99)
        self.assertEqual(len(hits), len(ROWS))

    def test_build_retriever_names(self) -> None:
        embedder = FakeEmbedder([1.0, 0.0, 0.0])
        for name in ("dense", "brute", "bm25", "hybrid"):
            self.assertEqual(build_retriever(name, self.collection, embedder).name, name)
        # hybrid-brute is a HybridRetriever, so it reports the hybrid name
        self.assertEqual(build_retriever("hybrid-brute", self.collection, embedder).name, "hybrid")
        with self.assertRaises(ValueError):
            build_retriever("magic", self.collection, embedder)

    def test_all_retrievers_satisfy_the_same_interface(self) -> None:
        """The point of the refactor: the harness must not be able to tell which
        strategy produced a result."""
        embedder = FakeEmbedder([1.0, 0.0, 0.0])
        for name in ("dense", "brute", "bm25", "hybrid", "hybrid-brute"):
            retriever = build_retriever(name, self.collection, embedder)
            hits = retriever.search("denda keterlambatan", 2)
            self.assertTrue(all(isinstance(h, Hit) for h in hits), name)
            self.assertLessEqual(len(hits), 2, name)
            self.assertTrue(retriever.score_label, name)


if __name__ == "__main__":
    unittest.main()
