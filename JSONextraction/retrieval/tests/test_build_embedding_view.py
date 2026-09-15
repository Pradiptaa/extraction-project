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
        # The fixture has no tables, so every row is a tree node.
        self.assertEqual(len(self.view["nodes"]), len(self.document["structure"]))
        self.assertEqual(self.view["node_count"], len(self.document["structure"]))
        self.assertEqual(self.view["structure_row_count"], len(self.document["structure"]))
        self.assertEqual(self.view["table_row_count"], 0)

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



def _with_tables(document: dict) -> dict:
    """The fixture plus an SSKK-shaped data sheet spanning two pages, modelled
    on the real p62-p63 sheet: page 12 holds tree nodes, page 13 holds none,
    and page 13's table has lost its header row to pdfplumber, which took the
    first data row as the header instead."""
    document = json.loads(json.dumps(document))
    document["tables"] = [
        {
            "table_id": "t_012_0", "page": 12,
            "headers": ["Pasal\ndalam\nSSUK", "Ketentuan", "Data"],
            "rows": [
                {"cells": ["28", "Penundaan", "Penundaan paling lama 14 hari"],
                 "refs_out": [{"raw": "28", "resolved_node_id": "n_0002", "type": "internal"}]},
                {"cells": ["", None, "  "], "refs_out": []},
                {"cells": ["99.9", "Tidak Ada", "Klausul yang tidak ada"],
                 "refs_out": [{"raw": "99.9", "resolved_node_id": None, "type": "internal"}]},
            ],
        },
        {
            "table_id": "t_013_0", "page": 13,
            "headers": ["", "Pedoman Pengoperasian", "diserahkan paling lambat ..... hari"],
            "rows": [{"cells": ["38.7", "Penyesuaian Harga", "tidak diberikan"], "refs_out": []}],
        },
        {
            # A genuinely repeated header: must not become a second row.
            "table_id": "t_013_1", "page": 13,
            "headers": ["Pasal dalam SSUK", "Ketentuan", "Data"],
            "rows": [{"cells": ["40.1", "Lain", "isi"], "refs_out": []}],
        },
    ]
    return document


class TableRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plain = build_embedding_view(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        self.view = build_embedding_view(_with_tables(json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))))
        self.table_rows = {"/".join(r["hierarchy_path"]): r for r in self.view["nodes"] if r["node_type"] == "table_row"}

    def test_counts_reconcile_per_source(self) -> None:
        # 5 data rows + 2 distinct header rows (t_012_0 real, t_013_0 is data),
        # minus 1 blank row. The repeated t_013_1 header is not emitted.
        self.assertEqual(self.view["structure_row_count"], 4)
        self.assertEqual(self.view["table_row_count"], 6)
        self.assertEqual(self.view["table_rows_skipped_empty"], 1)
        self.assertEqual(self.view["node_count"], 10)
        self.assertEqual(
            sorted(self.table_rows),
            ["t_012_0/0", "t_012_0/2", "t_012_0/h", "t_013_0/0", "t_013_0/h", "t_013_1/0"],
        )

    def test_adding_tables_leaves_tree_row_ids_unchanged(self) -> None:
        """What makes 2.1.0 additive, and `load --reuse-from` sound: a 2.0.0
        vector is addressed by an id this build still produces."""
        tree = [r["embedding_id"] for r in self.view["nodes"] if r["node_type"] != "table_row"]
        self.assertEqual(tree, [r["embedding_id"] for r in self.plain["nodes"]])

    def test_continuation_header_row_is_kept_as_data(self) -> None:
        """pdfplumber's 'header' on a continuation page is a data row that is
        NOT repeated in `rows`; dropping it would silently lose contract data."""
        self.assertIn("Pedoman Pengoperasian", self.table_rows["t_013_0/h"]["text"])

    def test_page_without_tree_nodes_inherits_the_last_sub_document(self) -> None:
        self.assertEqual(self.table_rows["t_013_0/0"]["sub_document"], "general_terms")
        self.assertEqual(self.table_rows["t_013_0/0"]["pages"], [13])

    def test_text_is_cells_without_header_labels(self) -> None:
        self.assertEqual(self.table_rows["t_012_0/0"]["text"], "28 | Penundaan | Penundaan paling lama 14 hari")

    def test_refs_carry_the_resolved_target_and_keep_unresolved_ones(self) -> None:
        resolved = self.table_rows["t_012_0/0"]["refs"]
        self.assertEqual(resolved[0]["target_sub_document"], "general_terms")
        self.assertEqual(resolved[0]["target_path"], ["28"])
        unresolved = self.table_rows["t_012_0/2"]["refs"]
        self.assertIsNone(unresolved[0]["target_path"])
        self.assertEqual(unresolved[0]["raw"], "99.9")

    def test_metadata_flattens_refs_for_chroma(self) -> None:
        from retrieval.load import _metadata

        self.assertEqual(_metadata(self.table_rows["t_012_0/0"])["ref_targets"], "general_terms:28")
        self.assertEqual(_metadata(self.table_rows["t_012_0/2"])["ref_targets"], "?:99.9")
        tree_row = next(r for r in self.view["nodes"] if r["node_type"] != "table_row")
        self.assertEqual(_metadata(tree_row)["ref_targets"], "")
        self.assertTrue(all(isinstance(v, (str, int, float, bool)) for v in _metadata(self.table_rows["t_012_0/0"]).values()))

    def test_all_ids_unique_with_tables(self) -> None:
        ids = [r["embedding_id"] for r in self.view["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
