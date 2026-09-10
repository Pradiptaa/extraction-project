"""Unit tests for retrieval.build_embedding_view.

Uses a small, committed synthetic fixture rather than a real
raw_extraction.json — those all land in output/, which is gitignored, so a
fresh checkout would have nothing for these tests to read.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from retrieval.build_embedding_view import build_embedding_view

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "sample_raw_extraction.json"


class BuildEmbeddingViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.view = build_embedding_view(self.document)

    def test_node_count_matches_raw_extraction(self) -> None:
        self.assertEqual(len(self.view["nodes"]), len(self.document["structure"]))
        self.assertEqual(self.view["node_count"], len(self.document["structure"]))

    def test_known_typo_survives_unmodified(self) -> None:
        # Fixture node n_0002 reproduces the source document's own typo
        # ("Pegawas" instead of "Pengawas") on purpose, mirroring a real
        # regression check in ground_truth/regression_checks.json
        # (bug_005_clause28_title) — the raw/embedding layers must never
        # silently "correct" a source typo.
        row = next(r for r in self.view["nodes"] if r["node_id"] == "n_0002")
        self.assertIn("Pegawas Pekerjaan", row["text"])
        self.assertNotIn("Pengawas Pekerjaan", row["text"])

    def test_known_identifier_survives_unmodified(self) -> None:
        row = next(r for r in self.view["nodes"] if r["node_id"] == "n_0003")
        self.assertIn("08/PUPRPRKP-B.PNK/SP-PPK", row["text"])

    def test_embedding_ids_are_unique(self) -> None:
        # Chroma keys rows on embedding_id, so any duplicate is silently
        # dropped on upsert rather than reported. Measured against the real
        # 6-specimen corpus, the pre-2.0.0 key collided on 48.6% of nodes.
        ids = [r["embedding_id"] for r in self.view["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_same_node_in_different_documents_gets_different_ids(self) -> None:
        # The specimens are all one standard form, so identical clause text
        # across two contracts is the norm, not an edge case. Ids must be
        # scoped by document or the second load overwrites the first.
        other = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        other["source"]["sha256"] = "different-document-hash"
        other_view = build_embedding_view(other)

        self.assertEqual(len(other_view["nodes"]), len(self.view["nodes"]))
        overlap = {r["embedding_id"] for r in self.view["nodes"]} & {
            r["embedding_id"] for r in other_view["nodes"]
        }
        self.assertEqual(overlap, set())

    def test_id_is_stable_for_identical_input(self) -> None:
        # The whole point of not using node_id: rebuilding an unchanged
        # document must reproduce the same ids, or every run re-embeds and
        # orphans the previous vectors.
        rebuilt = build_embedding_view(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        self.assertEqual(
            [r["embedding_id"] for r in rebuilt["nodes"]],
            [r["embedding_id"] for r in self.view["nodes"]],
        )

    def test_changed_text_changes_the_id(self) -> None:
        # A re-extraction that alters a node's text must produce a new id, so
        # the stale vector cannot stay silently attached to corrected text.
        mutated = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        mutated["structure"][1]["text_raw"] = "completely different body text"
        mutated_view = build_embedding_view(mutated)

        before = self.view["nodes"][1]["embedding_id"]
        after = mutated_view["nodes"][1]["embedding_id"]
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
