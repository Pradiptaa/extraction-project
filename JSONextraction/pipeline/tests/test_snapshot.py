"""`pipeline.snapshot` — normalization and the categorized diff.

    python -m unittest discover -s pipeline/tests -t .
"""
from __future__ import annotations

import copy
import unittest

from pipeline.snapshot import diff_snapshots, normalize


def _node(node_id, order, path, text="", title=None, node_type="clause", parent_id=None, depth=1, sub_document=None):
    return {
        "node_id": node_id, "parent_id": parent_id, "depth": depth, "node_type": node_type,
        "label": path[-1] if path else None, "label_normalized": path[-1] if path else None,
        "numbering_style": "decimal_plain" if path else None, "title": title, "path": path,
        "text_raw": text, "pages": [1], "reading_order": order, "sub_document": sub_document,
        "char_span": {"page": 1, "start": order * 10, "end": order * 10 + 5},
        "extraction": {"flags": []}, "refs_out": [],
    }


def _raw() -> dict:
    return {
        "schema_version": "1.0.0",
        "source": {"file": "x.pdf", "sha256": "abc", "page_count": 1, "pipeline_version": "1.0.0",
                   "extracted_at": "2026-01-01T00:00:00"},
        "profile": {"profile_id": "p", "match_score": 1.0},
        "core": {"contract_number": {"value": "01/SP/2024", "confidence": 0.9, "flags": [],
                                     "evidence": {"char_span": [5, 15]}}},
        "pages": [{"page": 1, "layout_type": "single_column", "sub_document": None,
                   "extraction_method": "native", "page_label": None, "raw_text": "t", "char_count": 1}],
        "structure": [
            _node("n_0001", 1, [], title="SURAT PERJANJIAN", node_type="heading", depth=0),
            _node("n_0002", 2, ["1"], "Pekerjaan dimulai.", "RUANG LINGKUP"),
            _node("n_0003", 3, ["2"], "Masa kontrak 90 hari.", "MASA KONTRAK"),
        ],
        "tables": [],
        "entities": [{"type": "nip", "raw": "NIP 1", "value": "1", "char_span": [0, 5]}],
        "quality": {"pipeline_status": "passed", "checks": [], "tree_quality_flags": []},
    }


class SnapshotDiffTests(unittest.TestCase):
    def diff(self, mutate) -> dict:
        after = _raw()
        mutate(after)
        return diff_snapshots(normalize(_raw(), "s"), normalize(after, "s")).changes

    def test_identical_extractions_diff_empty(self) -> None:
        self.assertTrue(diff_snapshots(normalize(_raw(), "s"), normalize(_raw(), "s")).empty)

    def test_volatile_fields_are_ignored(self) -> None:
        def mutate(raw):
            raw["source"]["extracted_at"] = "2030-01-01"
            raw["core"]["contract_number"]["evidence"] = {"char_span": [99, 109]}
            raw["entities"][0]["char_span"] = [50, 55]
            for i, n in enumerate(raw["structure"]):
                n["node_id"] = f"n_{i + 100:04d}"  # positional ids shift on any insertion
                n["char_span"] = {"page": 1, "start": 0, "end": 0}
        self.assertFalse(any(self.diff(mutate).values()))

    def test_edited_text_is_a_change_not_remove_plus_add(self) -> None:
        changes = self.diff(lambda raw: raw["structure"][2].update(text_raw="Masa kontrak 120 hari."))
        self.assertEqual(changes["structure"], [])
        self.assertEqual(len(changes["text"]), 1)

    def test_relabel_lands_in_labels_not_structure(self) -> None:
        changes = self.diff(lambda raw: raw["structure"][1].update(node_type="list_item", sub_document="x"))
        self.assertEqual(changes["structure"], [])
        self.assertEqual(len(changes["labels"]), 2)

    def test_reparenting_lands_in_structure(self) -> None:
        def mutate(raw):
            raw["structure"][2].update(parent_id="n_0002", depth=2)
        changes = self.diff(mutate)
        self.assertEqual(len(changes["structure"]), 2)  # parent_path and depth

    def test_removed_heading_is_reported_as_removed(self) -> None:
        changes = self.diff(lambda raw: raw["structure"].pop(0))
        self.assertEqual(len(changes["structure"]), 1)
        self.assertTrue(changes["structure"][0].startswith("- removed"))

    def test_unrelated_unnumbered_nodes_are_not_paired(self) -> None:
        def mutate(raw):
            raw["structure"][0].update(title="LAMPIRAN A DAFTAR KUANTITAS")
        changes = self.diff(mutate)
        self.assertEqual(sorted(line[0] for line in changes["structure"]), ["+", "-"])

    def test_core_and_entity_changes_are_categorized(self) -> None:
        def mutate(raw):
            raw["core"]["contract_number"]["value"] = "02/SP/2024"
            raw["entities"].append({"type": "nip", "raw": "NIP 2", "value": "2"})
        changes = self.diff(mutate)
        self.assertEqual(len(changes["core"]), 1)
        self.assertEqual(len(changes["entities"]), 1)

    def test_duplicate_entities_count_as_a_multiset(self) -> None:
        changes = self.diff(lambda raw: raw["entities"].append(copy.deepcopy(raw["entities"][0])))
        self.assertEqual(len(changes["entities"]), 1)


if __name__ == "__main__":
    unittest.main()
