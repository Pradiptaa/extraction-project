"""Behaviour of the retrieval regression gate itself.

Uses a hand-built collection with known vectors and a fake embedder, so the
harness is tested without an API key and without depending on what the real
model happens to think this week. The point is to pin the scoring RULES —
especially that a right-clause/wrong-document hit passes — not to measure
retrieval quality, which is what the live query set does.

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
    build_class_index,
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


# Two documents hold the same clause C/62 at slightly different vectors, which
# is the corpus's real shape: one standard form, many specimens.
ROWS = [
    ("a_62", [1.0, 0.0, 0.0, 0.0], _meta("aaaaaaaa", "general_terms", "C/62", "62"), "denda A"),
    ("b_62", [0.9, 0.1, 0.0, 0.0], _meta("bbbbbbbb", "general_terms", "C/62", "62"), "denda B"),
    ("a_41", [0.0, 1.0, 0.0, 0.0], _meta("aaaaaaaa", "general_terms", "B/41", "41"), "kahar A"),
    ("a_79", [0.0, 0.0, 1.0, 0.0], _meta("aaaaaaaa", "general_terms", "H/79", "79"), "sengketa A"),
    ("b_77", [0.0, 0.0, 0.0, 1.0], _meta("bbbbbbbb", "general_terms", "G/77", "77"), "cacat mutu B"),
    # A sub-clause of 41, the section node above it, and a clause whose path is
    # a string-prefix of another ("C/6" vs "C/61") — the three relations the
    # descendant rule has to tell apart.
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
        """The load-bearing rule. The query vector sits nearest document B's copy
        of clause 62; with k=1 only that copy is returned, and document A's copy
        is nowhere in the results. It must still pass — the clause is what was
        asked for, and every specimen is the same standard form."""
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
        """A near-miss on a different clause is a failure, so the lenient
        cross-document rule cannot be mistaken for 'anything passes'."""
        spec = _spec([_query("q_41", "B/41", "41")], default_k=1)
        ok, results, output = self._run(spec, [1.0, 0.0, 0.0, 0.0])

        self.assertFalse(ok)
        self.assertIsNone(results[0].rank)
        self.assertIn("nor any sub-clause in top-1", results[0].detail)
        self.assertIn("[FAIL]", output)
        # A failing query prints what it did return, so the miss is diagnosable
        # without a second run.
        self.assertIn("denda A", output)

    def test_rank_is_reported_and_max_rank_can_fail_a_hit(self) -> None:
        """Degradation inside top-k is the failure a boolean hides."""
        # k spans the whole fixture, so the hit is found and the test is about
        # where it ranks rather than about the cutoff.
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
        """The second leniency. The query lands on clause 41's sub-clause, not
        on 41 itself, which is the shape of almost every real hit: the heading
        is generic, the provision underneath it is what matches."""
        spec = _spec([_query("q_41", "B/41", "41")], default_k=1)
        ok, results, _ = self._run(spec, [0.0, 0.95, 0.05, 0.0])

        self.assertTrue(ok)
        self.assertEqual(results[0].rank, 1)
        self.assertIn("descendant", results[0].detail)
        self.assertIn("0 of them the clause itself", results[0].detail)

    def test_ancestor_hit_does_not_pass(self) -> None:
        """One-directional on purpose. Returning the whole section when asked
        about one clause is a precision loss, not a near-miss."""
        spec = _spec([_query("q_41", "B/41", "41")], default_k=1)
        ok, results, _ = self._run(spec, [0.0, 0.8, 0.2, 0.0])

        self.assertFalse(ok, "section B is an ancestor of B/41 and must not count")

    def test_path_prefix_does_not_leak_across_clause_numbers(self) -> None:
        """'C/6' is a string prefix of 'C/61' but not its ancestor. Without a
        segment boundary this would silently accept the wrong clause."""
        spec = _spec([_query("q_6", "C/6", "6")], default_k=1)
        ok, results, _ = self._run(spec, [0.0, 0.5, 0.0, 0.5])

        self.assertFalse(ok, "clause 61 must not satisfy an expectation of clause 6")

    def test_missing_clause_is_not_reported_as_a_retrieval_miss(self) -> None:
        """A stale expectation and a genuine retrieval failure need different
        fixes, so they must not produce the same message."""
        spec = _spec([_query("q_ghost", "Z/99", "99")], default_k=5)
        ok, results, _ = self._run(spec, [1.0, 0.0, 0.0, 0.0])

        self.assertFalse(ok)
        self.assertIn("not in the collection at all", results[0].detail)
        self.assertNotIn("top-", results[0].detail)

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
    """The shipped query set is itself ground truth, so it gets the same
    scrutiny as the extraction ground-truth files. Needs no API key and no
    Chroma — it is checked against the embedding views on disk."""

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
        """Catches a stale expectation at test time rather than as a mysterious
        run of failures against the live collection."""
        views = sorted(glob.glob(str(REPO / "output" / "embedding" / "*_embedding_view.json")))
        if not views:
            self.skipTest("no embedding views built")

        classes: dict[tuple[str, str, str], set[str]] = {}
        for path in views:
            view = json.loads(Path(path).read_text(encoding="utf-8"))
            for node in view["nodes"]:
                key = (
                    node.get("sub_document") or "",
                    "/".join(node.get("hierarchy_path") or []),
                    node.get("label_normalized") or "",
                )
                classes.setdefault(key, set()).add(view["source"]["file"])

        for query in load_queries(QUERY_SET)["queries"]:
            expect = query["expect"]
            key = (expect["sub_document"], expect["hierarchy_path"], expect["label"])
            with self.subTest(query=query["id"]):
                self.assertIn(key, classes, f"{query['id']} targets a clause no specimen contains")
                expected_documents = query.get("expect_documents")
                if expected_documents is not None:
                    self.assertEqual(
                        len(classes[key]), expected_documents,
                        f"{query['id']} recorded {expected_documents} documents, corpus now has "
                        f"{len(classes[key])} — update the query set if a specimen was added",
                    )


if __name__ == "__main__":
    unittest.main()
