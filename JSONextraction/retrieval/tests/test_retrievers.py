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
        self.scopes: list[set[str] | None] = []

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        self.scopes.append(scope)
        return [Hit(id=i, score=1.0, metadata={"label": i}, text=f"text {i}") for i in self.ids[:k]]


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [list(self.vector) for _ in texts]


# Two documents, with `r_dup` a near-identical copy of `r_denda` in the other
# one — the corpus's defining property in miniature, and what makes scoping
# testable: an unscoped search sees both copies, a scoped one must see exactly
# the copy belonging to its document.
DOC_A = "aaaa1111"
DOC_B = "bbbb2222"

ROWS = [
    ("r_denda", [1.0, 0.0, 0.0], "Pembayaran denda keterlambatan penyelesaian pekerjaan", DOC_A),
    ("r_kahar", [0.0, 1.0, 0.0], "Keadaan kahar force majeure", DOC_A),
    ("r_hki", [0.0, 0.0, 1.0], "Pelanggaran hak kekayaan intelektual oleh penyedia", DOC_A),
    ("r_dup", [0.9, 0.1, 0.0], "Pembayaran denda keterlambatan penyelesaian pekerjaan", DOC_B),
    # Filler, present for one reason: BM25Okapi's IDF turns 0 (and the row
    # becomes unretrievable, since a 0 score means "no opinion") once a term
    # occurs in half the corpus. With only the four rows above, "denda" sits in
    # 2 of 4 and the duplicate-selection tests below silently had nothing to
    # rank. Padding the corpus keeps document frequency low enough for the
    # lexical arm to behave as it does on the real 4522-row collection.
    ("r_pad_a1", [0.2, 0.3, 0.1], "Pengawas pekerjaan menerbitkan surat peringatan tertulis", DOC_A),
    ("r_pad_a2", [0.1, 0.2, 0.3], "Rapat persiapan pelaksanaan kontrak diselenggarakan", DOC_A),
    ("r_pad_b1", [0.3, 0.1, 0.2], "Jaminan pelaksanaan diserahkan sebelum penandatanganan", DOC_B),
    ("r_pad_b2", [0.2, 0.1, 0.3], "Penyesuaian harga satuan timpang tidak diberlakukan", DOC_B),
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
            metadatas=[
                {"label": r[0], "sub_document": "general_terms", "document_key": r[3]}
                for r in ROWS
            ],
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


class ScopeTests(RetrieverTests):
    """Document scoping, across every retriever.

    Inherits the fixture rather than rebuilding it, so a scoped search is
    always measured against the same corpus as the unscoped tests above.
    """

    ARMS = ("dense", "brute", "bm25", "hybrid", "hybrid-brute")

    def _retriever(self, name: str):
        return build_retriever(name, self.collection, FakeEmbedder([1.0, 0.0, 0.0]))

    def test_passing_no_scope_is_identical_to_passing_none(self) -> None:
        """The load-bearing regression test for this feature.

        `scope=None` must be exactly the pre-scoping behaviour, because the
        gate and every recorded baseline in `retrieval_baseline.json` were
        measured through the unscoped path. If these two ever diverge, the
        baselines silently stop describing what the gate measures.
        """
        for name in self.ARMS:
            retriever = self._retriever(name)
            default = [h.id for h in retriever.search("denda keterlambatan", 4)]
            explicit = [h.id for h in retriever.search("denda keterlambatan", 4, None)]
            self.assertEqual(default, explicit, name)

    def test_every_hit_comes_from_the_requested_document(self) -> None:
        for name in self.ARMS:
            hits = self._retriever(name).search("denda keterlambatan", 4, {DOC_B})
            self.assertTrue(hits, f"{name} returned nothing in scope")
            for hit in hits:
                self.assertEqual(hit.metadata.get("document_key"), DOC_B, name)

    def test_scope_selects_between_byte_identical_copies(self) -> None:
        """`r_denda` and `r_dup` hold the same text in different documents.

        Unscoped, either may come back; scoped, only the requested document's
        copy may. This is the whole point of the feature on a corpus that is
        six copies of one standard form.
        """
        for name in self.ARMS:
            retriever = self._retriever(name)
            self.assertEqual(
                [h.id for h in retriever.search("denda keterlambatan", 1, {DOC_A})], ["r_denda"], name
            )
            self.assertEqual(
                [h.id for h in retriever.search("denda keterlambatan", 1, {DOC_B})], ["r_dup"], name
            )

    def test_scope_can_name_several_documents(self) -> None:
        for name in self.ARMS:
            ids = {h.id for h in self._retriever(name).search("denda keterlambatan", 9, {DOC_A, DOC_B})}
            self.assertIn("r_denda", ids, name)
            self.assertIn("r_dup", ids, name)

    def test_scope_matching_nothing_returns_nothing(self) -> None:
        """Never a silent fallback to the whole corpus: answering from six
        contracts when one was asked for is the failure this flag prevents."""
        for name in self.ARMS:
            self.assertEqual(self._retriever(name).search("denda", 5, {"no_such_document"}), [], name)

    def test_scoped_dense_matches_scoped_brute(self) -> None:
        """Dense is the only arm whose scope is applied inside the index, so it
        is the only one that could lose recall to filtering. Brute force is the
        exact reference — the same comparison §7 requires before any claim
        about ranking."""
        embedder = FakeEmbedder([1.0, 0.0, 0.0])
        dense = DenseRetriever(self.collection, embedder).search("denda", 3, {DOC_A})
        brute = BruteForceRetriever(self.collection, embedder).search("denda", 3, {DOC_A})
        # Compared by DISTANCE, not by id. Equally close rows are equally good
        # answers, and this fixture has exact ties (r_kahar and r_hki are both
        # orthogonal to the query), so an id comparison would be measuring
        # arbitrary tie-breaking — the same trap §7 records on the real corpus.
        self.assertEqual(len(dense), len(brute))
        for dense_hit, brute_hit in zip(dense, brute):
            self.assertAlmostEqual(dense_hit.score, brute_hit.score, places=5)

    def test_hybrid_scopes_both_sides_rather_than_filtering_the_fused_list(self) -> None:
        """Filtering after fusion would spend the pool on out-of-scope rows and
        routinely return nothing on this corpus."""
        dense, lexical = StaticRetriever(["a", "b"]), StaticRetriever(["b", "c"])
        HybridRetriever(dense, lexical, pool=7).search("q", 3, {DOC_A})
        self.assertEqual(dense.scopes, [{DOC_A}])
        self.assertEqual(lexical.scopes, [{DOC_A}])

    def test_bm25_idf_stays_corpus_wide_under_a_scope(self) -> None:
        """Strategy A, pinned: a scope narrows the candidates, never the
        statistics. If BM25 were rebuilt per scope, IDF would be computed within
        one document and the same row would score differently — which would make
        scoped and unscoped results incomparable."""
        retriever = self._retriever("bm25")
        unscoped = {h.id: h.score for h in retriever.search("denda keterlambatan", 9)}
        scoped = retriever.search("denda keterlambatan", 9, {DOC_A})
        for hit in scoped:
            self.assertAlmostEqual(hit.score, unscoped[hit.id], places=9)


if __name__ == "__main__":
    unittest.main()
