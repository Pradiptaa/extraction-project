"""Behaviour of the retrieval regression gate itself. A hand-built collection
with known vectors and a fake embedder pins the scoring rules, not retrieval
quality — that is what the live query set measures.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import glob
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import chromadb

from retrieval.config import collection_name
from retrieval.retrieval_evaluate import (
    DEFAULT_BASELINE,
    EQUIVALENCE_MIN_CHARS,
    QueryResult,
    accepted_rows,
    ref_check,
    baseline_key,
    compare_to_baseline,
    load_baseline,
    write_baseline,
    build_class_index,
    evaluate_query,
    clause_key,
    load_queries,
    open_collection,
    run,
)
from retrieval.schema import EMBEDDING_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]
QUERY_SET = REPO / "ground_truth" / "retrieval_queries.json"


def _meta(document_key: str, sub: str, path: str, label: str) -> dict:
    return {
        "document_key": document_key,
        "node_id": f"n_{label}",
        "node_type": "clause",
        "sub_document": sub,
        "label": label,
        "hierarchy_path": path,
        "depth": 2,
        "page_first": 1,
        "page_last": 1,
        "schema_version": EMBEDDING_SCHEMA_VERSION,
    }


# Two documents hold the same clause C/62 at slightly different vectors — the
# corpus's real shape: one standard form, many specimens.
ROWS = [
    ("a_62", [1.0, 0.0, 0.0, 0.0], _meta("aaaaaaaa", "general_terms", "C/62", "62"), "denda A"),
    ("b_62", [0.9, 0.1, 0.0, 0.0], _meta("bbbbbbbb", "general_terms", "C/62", "62"), "denda B"),
    ("a_41", [0.0, 1.0, 0.0, 0.0], _meta("aaaaaaaa", "general_terms", "B/41", "41"), "kahar A"),
    ("a_79", [0.0, 0.0, 1.0, 0.0], _meta("aaaaaaaa", "general_terms", "H/79", "79"), "sengketa A"),
    ("b_77", [0.0, 0.0, 0.0, 1.0], _meta("bbbbbbbb", "general_terms", "G/77", "77"), "cacat mutu B"),
    # A sub-clause, its section node, and a string-prefix clause ("C/6" vs
    # "C/61") — the three relations the descendant rule must tell apart.
    ("a_41_2", [0.0, 0.95, 0.05, 0.0], _meta("aaaaaaaa", "general_terms", "B/41/41.2", "41.2"), "kahar rincian"),
    ("a_B", [0.0, 0.8, 0.2, 0.0], _meta("aaaaaaaa", "general_terms", "B", "B"), "section B"),
    ("a_6", [0.5, 0.0, 0.0, 0.5], _meta("aaaaaaaa", "general_terms", "C/6", "6"), "kkn A"),
    ("a_61", [0.0, 0.5, 0.0, 0.5], _meta("aaaaaaaa", "general_terms", "C/61", "61"), "alih keahlian A"),
]


class VectorEmbedder:
    """Returns a preset vector for any query, so ranking is fully determined."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [list(self.vector) for _ in texts]


def _spec(queries: list[dict], default_k: int = 5) -> dict:
    return {
        "schema_version": "1.0.0",
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "default_k": default_k,
        "queries": queries,
    }


def _query(query_id: str, path: str, label: str, **extra) -> dict:
    spec = {
        "id": query_id,
        "query": f"query for {path}",
        "expect": {"sub_document": "general_terms", "hierarchy_path": path, "label": label},
    }
    spec.update(extra)
    return spec


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        client = chromadb.PersistentClient(path=str(self.tmp / "chroma"))
        self.collection = client.get_or_create_collection(
            collection_name("test", "fake-model"), metadata={"hnsw:space": "cosine"}
        )
        self.collection.add(
            ids=[r[0] for r in ROWS],
            embeddings=[r[1] for r in ROWS],
            metadatas=[r[2] for r in ROWS],
            documents=[r[3] for r in ROWS],
        )

    def _run(self, spec, vector, **kwargs):
        embedder = VectorEmbedder(vector)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ok, results = run(spec, self.collection, embedder, spec["default_k"], **kwargs)
        return ok, results, buffer.getvalue()

    def test_hit_in_the_other_document_still_passes(self) -> None:
        """With k=1 only document B's copy of clause 62 is returned. It must
        still pass — the clause is what was asked for."""
        spec = _spec([_query("q_62", "C/62", "62")], default_k=1)
        ok, results, _ = self._run(spec, [0.9, 0.1, 0.0, 0.0])

        self.assertTrue(ok)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].documents_hit, ["bbbbbbbb"])
        self.assertEqual(results[0].class_size, 2, "both documents' copies form one class")

    def test_both_copies_counted_when_k_is_wide(self) -> None:
        spec = _spec([_query("q_62", "C/62", "62")], default_k=3)
        _, results, _ = self._run(spec, [1.0, 0.0, 0.0, 0.0])

        self.assertEqual(results[0].rank, 1)
        self.assertCountEqual(results[0].documents_hit, ["aaaaaaaa", "bbbbbbbb"])

    def test_wrong_clause_fails(self) -> None:
        """So the lenient cross-document rule cannot mean "anything passes"."""
        spec = _spec([_query("q_41", "B/41", "41")], default_k=1)
        ok, results, output = self._run(spec, [1.0, 0.0, 0.0, 0.0])

        self.assertFalse(ok)
        self.assertIsNone(results[0].rank)
        self.assertIn("nor any sub-clause in top-1", results[0].detail)
        self.assertIn("[FAIL]", output)
        # A failing query prints what it returned, so the miss is diagnosable.
        self.assertIn("denda A", output)

    def test_rank_is_reported_and_max_rank_can_fail_a_hit(self) -> None:
        """Degradation inside top-k is the failure a boolean hides."""
        # k spans the fixture, so this is about rank, not the cutoff.
        spec = _spec([_query("q_77", "G/77", "77")], default_k=len(ROWS))

        ok, results, _ = self._run(spec, [0.6, 0.5, 0.4, 0.3])
        self.assertTrue(ok)
        rank = results[0].rank
        self.assertIsNotNone(rank)
        self.assertGreater(rank, 1, "this query deliberately does not rank the target first")

        ok, results, _ = self._run(spec, [0.6, 0.5, 0.4, 0.3], max_rank=rank - 1)
        self.assertFalse(ok, "a rank worse than --max-rank must fail")
        self.assertEqual(results[0].rank, rank, "the rank is still reported on a max-rank failure")
        self.assertIn("--max-rank", results[0].detail)

    def test_sub_clause_hit_passes_and_is_labelled_as_a_descendant(self) -> None:
        """The shape of almost every real hit: the heading is generic, and the
        provision underneath it is what matches."""
        spec = _spec([_query("q_41", "B/41", "41")], default_k=1)
        ok, results, _ = self._run(spec, [0.0, 0.95, 0.05, 0.0])

        self.assertTrue(ok)
        self.assertEqual(results[0].rank, 1)
        self.assertIn("descendant", results[0].detail)
        self.assertIn("0 of them the clause itself", results[0].detail)

    def test_ancestor_hit_does_not_pass(self) -> None:
        """Returning a whole section is a precision loss, not a near-miss."""
        spec = _spec([_query("q_41", "B/41", "41")], default_k=1)
        ok, results, _ = self._run(spec, [0.0, 0.8, 0.2, 0.0])

        self.assertFalse(ok, "section B is an ancestor of B/41 and must not count")

    def test_path_prefix_does_not_leak_across_clause_numbers(self) -> None:
        """'C/6' is a string prefix of 'C/61' but not its ancestor."""
        spec = _spec([_query("q_6", "C/6", "6")], default_k=1)
        ok, results, _ = self._run(spec, [0.0, 0.5, 0.0, 0.5])

        self.assertFalse(ok, "clause 61 must not satisfy an expectation of clause 6")

    def test_missing_clause_is_not_reported_as_a_retrieval_miss(self) -> None:
        """A stale expectation and a real miss need different fixes."""
        spec = _spec([_query("q_ghost", "Z/99", "99")], default_k=5)
        ok, results, _ = self._run(spec, [1.0, 0.0, 0.0, 0.0])

        self.assertFalse(ok)
        self.assertIn("not in the collection at all", results[0].detail)
        self.assertNotIn("top-", results[0].detail)

    def test_byte_identical_text_under_another_clause_key_passes(self) -> None:
        """`x_62` carries clause 62's exact text under the letterless path —
        the real clause-path inconsistency. Without this rule, passing depended
        on which of two identical rows the retriever happened to return."""
        self.collection.add(
            ids=["x_62"],
            embeddings=[[0.99, 0.01, 0.0, 0.0]],
            metadatas=[_meta("cccccccc", "general_terms", "62", "62")],
            documents=["denda A"],  # byte-identical to a_62
        )
        spec = _spec([_query("q_62", "C/62", "62")], default_k=1)
        ok, results, _ = self._run(spec, [0.99, 0.01, 0.0, 0.0])

        self.assertTrue(ok, "a row holding the expected clause's exact text is the same answer")
        self.assertIn("equivalent", results[0].detail)

    def test_equivalence_is_reported_separately_from_an_exact_hit(self) -> None:
        """A row qualifying only by text equality is a weaker kind of hit."""
        index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "C/62", "label": "62"}
        self.assertEqual(set(accepted_rows(index, expect).values()), {"exact"})

    def test_equivalence_does_not_widen_to_different_text(self) -> None:
        """Widening is by exact text equality only."""
        self.collection.add(
            ids=["y_62"],
            embeddings=[[0.98, 0.02, 0.0, 0.0]],
            metadatas=[_meta("dddddddd", "general_terms", "62", "62")],
            documents=["denda A but with more words appended"],
        )
        index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "C/62", "label": "62"}
        self.assertNotIn("y_62", accepted_rows(index, expect))

    def test_equivalence_does_not_rescue_an_ancestor(self) -> None:
        """Equivalence widens by text, so it must not become a back door for a
        relation the clause rules deliberately reject."""
        self.collection.add(
            ids=["z_B"],
            embeddings=[[0.0, 0.79, 0.21, 0.0]],
            metadatas=[_meta("eeeeeeee", "general_terms", "B", "B")],
            documents=["section B"],  # identical to a_B, which is an ancestor of B/41
        )
        index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "B/41", "label": "41"}
        accepted = accepted_rows(index, expect)
        self.assertNotIn("a_B", accepted)
        self.assertNotIn("z_B", accepted)

    def test_short_identical_text_under_an_unrelated_clause_is_not_equivalent(self) -> None:
        """Short strings recur verbatim under unrelated clauses, so text
        equality alone would let one pass a query about a different clause."""
        self.collection.add(
            ids=["frag_41_child", "frag_elsewhere"],
            embeddings=[[0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1]],
            metadatas=[
                _meta("aaaaaaaa", "general_terms", "B/41/41.3", "41.3"),
                _meta("bbbbbbbb", "general_terms", "H/79/79.1", "79.1"),
            ],
            documents=["Pengadilan.", "Pengadilan."],
        )
        index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "B/41", "label": "41"}
        accepted = accepted_rows(index, expect)
        self.assertEqual(accepted.get("frag_41_child"), "descendant")
        self.assertNotIn("frag_elsewhere", accepted)

    def test_long_identical_text_under_a_broken_path_is_still_equivalent(self) -> None:
        """One specimen files clause sentences under a malformed path; a
        sentence that long and identical is still the same provision."""
        sentence = "Tidak termasuk Keadaan Kahar adalah hal-hal merugikan yang disebabkan oleh perbuatan para pihak."
        self.assertGreaterEqual(len(sentence), EQUIVALENCE_MIN_CHARS)
        self.collection.add(
            ids=["long_41_child", "long_broken"],
            embeddings=[[0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1]],
            metadatas=[
                _meta("aaaaaaaa", "general_terms", "B/41/41.4", "41.4"),
                _meta("bbbbbbbb", "general_terms", "B/B.5/1.120", "1.120"),
            ],
            documents=[sentence, sentence],
        )
        index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "B/41", "label": "41"}
        self.assertEqual(accepted_rows(index, expect).get("long_broken"), "equivalent")

    def _add_sskk(self) -> None:
        """An SSKK table row and the SSUK clause sharing its topic word."""
        sskk = dict(_meta("aaaaaaaa", "special_terms", "t_062_0/0", ""), node_type="table_row",
                    ref_targets="general_terms:A/4/4.1;general_terms:A/4/4.2")
        sskk_other_doc = dict(_meta("bbbbbbbb", "special_terms", "t_054_0/0", ""), node_type="table_row",
                              ref_targets="general_terms:4/4.1")
        broken = dict(_meta("cccccccc", "special_terms", "t_055_0/0", ""), node_type="table_row",
                      ref_targets="?:4.1")
        ssuk = dict(_meta("aaaaaaaa", "general_terms", "A/4", "4"), node_type="clause")
        self.collection.add(
            ids=["sskk_a", "sskk_b", "sskk_broken", "ssuk_4"],
            embeddings=[[0.0, 0.0, 0.7, 0.7], [0.0, 0.0, 0.69, 0.71], [0.0, 0.0, 0.6, 0.8], [0.0, 0.1, 0.7, 0.7]],
            metadatas=[sskk, sskk_other_doc, broken, ssuk],
            documents=["4.1 & 4.2 | Korespondensi | Alamat A", "4.1 & 4.2 | Korespondensi | Alamat B",
                       "4.1 & 4.2 | Korespondensi | Alamat C", "Korespondensi"],
        )

    def test_content_target_accepts_matching_rows_in_any_document(self) -> None:
        self._add_sskk()
        index = build_class_index(self.collection)
        expect = {"sub_document": "special_terms", "node_type": "table_row", "text_contains": "| Korespondensi |"}
        self.assertEqual(accepted_rows(index, expect), {"sskk_a": "exact", "sskk_b": "exact", "sskk_broken": "exact"})

    def test_node_type_filter_keeps_the_same_topic_clause_out(self) -> None:
        """The SSUK heading shares the word but holds no addresses."""
        self._add_sskk()
        index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "A/4", "label": "4"}
        self.assertIn("ssuk_4", accepted_rows(index, expect))
        self.assertNotIn("ssuk_4", accepted_rows(index, dict(expect, node_type="table_row")))

    def test_text_contains_narrows_a_clause_target_byte_exactly(self) -> None:
        """Identifier/typo survival: the right clause must also carry the exact
        string, with no normalisation."""
        self.collection.add(
            ids=["typo", "corrected"],
            embeddings=[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            metadatas=[_meta("aaaaaaaa", "general_terms", "B/28", "28"), _meta("bbbbbbbb", "general_terms", "B/28", "28")],
            documents=["Penundaan Oleh Pegawas Pekerjaan", "Penundaan Oleh Pengawas Pekerjaan"],
        )
        index = build_class_index(self.collection)
        accepted = accepted_rows(index, {"sub_document": "general_terms", "hierarchy_path": "B/28",
                                         "label": "28", "text_contains": "Pegawas Pekerjaan"})
        self.assertEqual(set(accepted), {"typo"})

    def test_ref_check_matches_path_suffix_on_a_segment_boundary(self) -> None:
        expect_ref = {"sub_document": "general_terms", "path_suffix": "4/4.1"}
        self.assertTrue(ref_check({"ref_targets": "general_terms:A/4/4.1"}, expect_ref))
        self.assertTrue(ref_check({"ref_targets": "x:1;general_terms:4/4.1"}, expect_ref), "letterless path")
        self.assertFalse(ref_check({"ref_targets": "general_terms:A/14/4.1"}, expect_ref), "14/4.1 is not 4/4.1")
        self.assertFalse(ref_check({"ref_targets": "main_agreement:A/4/4.1"}, expect_ref), "wrong namespace")
        self.assertFalse(ref_check({"ref_targets": "?:4.1"}, expect_ref), "unresolved")
        self.assertFalse(ref_check({}, expect_ref))

    def test_retrieved_row_with_a_broken_reference_fails_the_query(self) -> None:
        """Finding the row is not enough if its cross-reference no longer
        resolves to the SSUK clause."""
        self._add_sskk()
        query = {
            "id": "q_sskk", "query": "korespondensi",
            "expect": {"sub_document": "special_terms", "node_type": "table_row", "text_contains": "| Korespondensi |"},
            "expect_ref": {"sub_document": "general_terms", "path_suffix": "4/4.1"},
        }
        ok, results, _ = self._run(_spec([query], default_k=1), [0.0, 0.0, 0.6, 0.8])  # nearest: sskk_broken
        self.assertFalse(ok)
        self.assertIn("no accepted hit references", results[0].detail)
        self.assertIn("?:4.1", results[0].detail)

        ok, results, _ = self._run(_spec([query], default_k=1), [0.0, 0.0, 0.7, 0.7])  # nearest: sskk_a
        self.assertTrue(ok, results[0].detail)

    def test_score_does_not_depend_on_how_ties_are_ordered(self) -> None:
        """What makes this gate usable for comparing retrievers: a fake
        retriever returns the same equal-scoring rows in every possible order,
        and the verdict must not move."""
        from itertools import permutations

        from retrieval.retrievers import Hit

        # Identical text under the expected clause, the letterless path, and
        # another document.
        self.collection.add(
            ids=["tie_plain", "tie_other_doc"],
            embeddings=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            metadatas=[
                _meta("cccccccc", "general_terms", "62", "62"),
                _meta("dddddddd", "general_terms", "C/62", "62"),
            ],
            documents=["denda A", "denda A"],
        )
        class_index = build_class_index(self.collection)
        expect = {"sub_document": "general_terms", "hierarchy_path": "C/62", "label": "62"}
        tied = ["a_62", "tie_plain", "tie_other_doc"]

        verdicts = set()
        for order in permutations(tied):
            hits = [Hit(id=i, score=0.5, metadata={}, text="denda A") for i in order]
            retriever = type("R", (), {"search": lambda self, q, k, h=hits: h[:k]})()
            result = evaluate_query(
                {"id": "q", "query": "denda", "expect": expect},
                retriever, class_index, 1, None,
            )
            verdicts.add(result.passed)

        self.assertEqual(verdicts, {True}, "verdict moved with tie order alone")

    def test_per_query_k_overrides_the_default(self) -> None:
        spec = _spec([_query("q_77", "G/77", "77", k=2)], default_k=5)
        ok, _, _ = self._run(spec, [0.6, 0.5, 0.4, 0.3])
        self.assertFalse(ok, "k=2 should exclude a rank-4 hit that k=5 accepts")

    def test_summary_line_and_class_index(self) -> None:
        spec = _spec([_query("q_62", "C/62", "62"), _query("q_41", "B/41", "41")], default_k=3)
        ok, _, output = self._run(spec, [1.0, 0.0, 0.0, 0.0])

        self.assertFalse(ok)
        self.assertIn("1/2 retrieval checks passed", output)
        self.assertIn("RESULT: FAIL", output)

        index = build_class_index(self.collection)
        self.assertEqual(len(index[("general_terms", "C/62", "62")]), 2)
        self.assertEqual(clause_key(ROWS[0][2]), ("general_terms", "C/62", "62"))

    def test_baseline_verdict_passes_at_baseline_and_fails_on_regression(self) -> None:
        """Red only when something that used to pass stops passing."""
        spec = _spec([_query("q_62", "C/62", "62"), _query("q_41", "B/41", "41")], default_k=1)
        vector = [1.0, 0.0, 0.0, 0.0]  # q_62 passes, q_41 fails

        ok, _, output = self._run(spec, vector, expected_pass={"q_62"})
        self.assertTrue(ok, "a known failure recorded in the baseline is not a regression")
        self.assertIn("RESULT: PASS (no regressions", output)

        ok, _, output = self._run(spec, vector, expected_pass={"q_62", "q_41"})
        self.assertFalse(ok)
        self.assertIn("[REGRESSION] q_41", output)

        ok, _, output = self._run(spec, vector, expected_pass=set())
        self.assertTrue(ok, "a new pass never fails the run")
        self.assertIn("[NEW PASS]   q_62", output)

    def test_empty_collection_is_refused_rather_than_failing_every_query(self) -> None:
        from retrieval.config import Settings

        empty_dir = self.tmp / "empty"
        settings = Settings(
            api_key="fake", model="fake-model", batch_size=2, request_delay=0.0,
            db_path=empty_dir, collection=collection_name("test", "fake-model"),
        )
        with self.assertRaises(SystemExit) as caught:
            open_collection(settings)
        self.assertIn("run `python -m retrieval.load`", str(caught.exception))

        chromadb.PersistentClient(path=str(empty_dir)).get_or_create_collection(settings.collection)
        with self.assertRaises(SystemExit) as caught:
            open_collection(settings)
        self.assertIn("is empty", str(caught.exception))


class QuerySetTests(unittest.TestCase):
    """The shipped query set is itself ground truth, checked against the
    embedding views on disk rather than Chroma."""

    def test_query_set_is_well_formed(self) -> None:
        spec = load_queries(QUERY_SET)
        self.assertGreaterEqual(len(spec["queries"]), 10)
        self.assertLessEqual(len(spec["queries"]), 20)

    def test_schema_mismatch_is_refused(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)

        spec = json.loads(QUERY_SET.read_text(encoding="utf-8"))
        spec["embedding_schema_version"] = "1.0.0"
        stale = tmp / "stale.json"
        stale.write_text(json.dumps(spec), encoding="utf-8")

        with self.assertRaises(SystemExit) as caught:
            load_queries(stale)
        self.assertIn("1.0.0", str(caught.exception))

    def test_query_without_a_target_is_refused(self) -> None:
        spec = _spec([{"id": "q", "query": "x", "expect": {"sub_document": "general_terms"}}])
        path = Path(tempfile.mkdtemp()) / "q.json"
        self.addCleanup(shutil.rmtree, path.parent, ignore_errors=True)
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            load_queries(path)
        self.assertIn("hierarchy_path or text_contains", str(caught.exception))

    def test_duplicate_query_id_is_refused(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)

        spec = json.loads(QUERY_SET.read_text(encoding="utf-8"))
        spec["queries"].append(dict(spec["queries"][0]))
        duped = tmp / "duped.json"
        duped.write_text(json.dumps(spec), encoding="utf-8")

        with self.assertRaises(SystemExit) as caught:
            load_queries(duped)
        self.assertIn("duplicate query id", str(caught.exception))

    def test_every_expected_clause_exists_in_the_corpus(self) -> None:
        """Catches a stale expectation at test time rather than as a run of
        failures against the live collection. Uses the gate's own
        `accepted_rows`, so targets are checked by the rule that will score
        them; `expect_documents` counts only exact rows."""
        views = sorted(glob.glob(str(REPO / "output" / "embedding" / "*_embedding_view.json")))
        if not views:
            self.skipTest("no embedding views built")

        class_index: dict = {}
        for path in views:
            view = json.loads(Path(path).read_text(encoding="utf-8"))
            if view.get("schema_version") != EMBEDDING_SCHEMA_VERSION:
                self.skipTest(f"embedding views on disk are schema {view.get('schema_version')}, rebuild them")
            for node in view["nodes"]:
                entry = {
                    "_id": node["embedding_id"],
                    "_text": node.get("text") or "",
                    "_file": view["source"]["file"],
                    "sub_document": node.get("sub_document") or "",
                    "hierarchy_path": "/".join(node.get("hierarchy_path") or []),
                    "label": node.get("label_normalized") or "",
                    "node_type": node.get("node_type") or "",
                }
                class_index.setdefault(clause_key(entry), []).append(entry)
        files = {row["_id"]: row["_file"] for rows in class_index.values() for row in rows}

        for query in load_queries(QUERY_SET)["queries"]:
            accepted = accepted_rows(class_index, query["expect"])
            with self.subTest(query=query["id"]):
                self.assertTrue(accepted, f"{query['id']} targets something no specimen contains")
                expected_documents = query.get("expect_documents")
                if expected_documents is not None:
                    documents = {files[i] for i, kind in accepted.items() if kind == "exact"}
                    self.assertEqual(
                        len(documents), expected_documents,
                        f"{query['id']} recorded {expected_documents} documents, corpus now has "
                        f"{len(documents)} — update the query set if a specimen was added",
                    )




class BaselineTests(unittest.TestCase):
    """The baseline turns an always-red gate into a regression check, so the
    rules for when it applies are pinned here."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = self.tmp / "baseline.json"

    @staticmethod
    def _results(**passed: bool) -> list[QueryResult]:
        return [QueryResult(query_id=q, query=q, passed=p, rank=None, detail="") for q, p in passed.items()]

    def test_compare_reports_regressions_and_improvements(self) -> None:
        results = self._results(q1=True, q2=False, q3=True)
        self.assertEqual(compare_to_baseline(results, {"q1", "q2"}), (["q2"], ["q3"]))

    def test_query_removed_from_the_set_is_not_a_regression(self) -> None:
        self.assertEqual(compare_to_baseline(self._results(q1=True), {"q1", "gone"}), ([], []))

    def test_key_includes_tokenizer_only_where_it_matters(self) -> None:
        self.assertEqual(baseline_key("dense", "stem", 5), "dense/k=5")
        self.assertEqual(baseline_key("hybrid", "plain", 5), "hybrid/plain/k=5")
        self.assertNotEqual(baseline_key("bm25", "plain", 5), baseline_key("bm25", "plain", 10))

    def test_round_trip(self) -> None:
        write_baseline(self.path, "dense/k=5", "coll", self._results(q1=True, q2=False), "2026-09-15")
        self.assertEqual(load_baseline(self.path, "dense/k=5", "coll"), {"q1"})

    def test_baseline_from_another_collection_does_not_apply(self) -> None:
        """A different model, schema or index is a different system."""
        write_baseline(self.path, "dense/k=5", "old_coll", self._results(q1=True), "2026-09-15")
        with self.assertLogs("retrieval.evaluate", level="WARNING"):
            self.assertIsNone(load_baseline(self.path, "dense/k=5", "new_coll"))

    def test_missing_file_or_entry_falls_back_to_strict(self) -> None:
        with self.assertLogs("retrieval.evaluate", level="WARNING"):
            self.assertIsNone(load_baseline(self.path, "dense/k=5", "coll"))
        write_baseline(self.path, "dense/k=5", "coll", self._results(q1=True), "2026-09-15")
        with self.assertLogs("retrieval.evaluate", level="WARNING"):
            self.assertIsNone(load_baseline(self.path, "bm25/plain/k=5", "coll"))

    def test_shipped_baseline_names_only_real_queries(self) -> None:
        """A baseline id that is not in the query set can never regress, so it
        would silently weaken the gate."""
        if not DEFAULT_BASELINE.exists():
            self.skipTest("no shipped baseline")
        query_ids = {q["id"] for q in load_queries(QUERY_SET)["queries"]}
        spec = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
        for key, entry in spec["baselines"].items():
            self.assertLessEqual(set(entry["passed"]), query_ids, key)


if __name__ == "__main__":
    unittest.main()
