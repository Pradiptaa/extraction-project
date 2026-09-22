"""`pipeline.structure_score` — boundary F1 and parent accuracy.

The metric's job is to stay blind to labels while reacting to shape, so both
halves are tested: a relabelling must score 1.0, and a merge, a split or a
reparenting must not.
"""
from __future__ import annotations

import unittest

from pipeline.snapshot import normalize
from pipeline.structure_score import score
from pipeline.tests.test_snapshot import _node, _raw


def _nodes(raw: dict) -> list[dict]:
    return normalize(raw, "s")["nodes"]


class StructureScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = _nodes(_raw())

    def test_identical_trees_score_one(self) -> None:
        p, r, f1, parent, depth, matched = score(self.reference, _nodes(_raw()))
        self.assertEqual((f1, parent, depth), (1.0, 1.0, 1.0))
        self.assertEqual(matched, len(self.reference))

    def test_relabelling_does_not_move_the_score(self) -> None:
        """The Phase 3 engine swap changes node_type and sub_document on
        purpose; the metric must ignore exactly that."""
        raw = _raw()
        for n in raw["structure"]:
            n["node_type"] = "list_item"
            n["sub_document"] = "somewhere_else"
            n["title"] = n["title"]
        _, _, f1, parent, _, _ = score(self.reference, _nodes(raw))
        self.assertEqual((f1, parent), (1.0, 1.0))

    def test_merged_nodes_lower_boundary_f1(self) -> None:
        raw = _raw()
        raw["structure"][1]["text_raw"] += " " + raw["structure"][2]["text_raw"]
        del raw["structure"][2]
        _, recall, f1, _, _, _ = score(self.reference, _nodes(raw))
        self.assertLess(f1, 1.0)
        self.assertLess(recall, 1.0)

    def test_split_node_lowers_precision(self) -> None:
        raw = _raw()
        raw["structure"][1]["text_raw"] = "Pekerjaan"
        raw["structure"].append(_node("n_0004", 4, ["1", "a"], "dimulai.", parent_id="n_0002", depth=2))
        precision, _, f1, _, _, _ = score(self.reference, _nodes(raw))
        self.assertLess(precision, 1.0)
        self.assertLess(f1, 1.0)

    def test_reparenting_lowers_parent_accuracy_only(self) -> None:
        raw = _raw()
        raw["structure"][2]["parent_id"] = "n_0002"
        raw["structure"][2]["path"] = ["1", "2"]
        _, _, f1, parent, _, _ = score(self.reference, _nodes(raw))
        self.assertEqual(f1, 1.0)          # same text units
        self.assertLess(parent, 1.0)       # under a different parent

    def test_empty_headings_are_kept_distinct(self) -> None:
        """Text-only identity would make every bare heading the same unit, so a
        lost heading would be invisible."""
        raw = _raw()
        raw["structure"].append(_node("n_0005", 5, [], title="LAMPIRAN A", node_type="heading", depth=0))
        precision, _, _, _, _, _ = score(self.reference, _nodes(raw))
        self.assertLess(precision, 1.0)


if __name__ == "__main__":
    unittest.main()
