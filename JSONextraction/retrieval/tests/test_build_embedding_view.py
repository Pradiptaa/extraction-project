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


if __name__ == "__main__":
    unittest.main()
