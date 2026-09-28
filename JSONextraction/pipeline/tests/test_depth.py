"""`pipeline.depth` — the relative-depth engine (fix_plan Phase 3).

Each test is a nesting order the fixed `_BASE_DEPTH` table could not express, or
a hazard found while building it on the real specimens.
"""
from __future__ import annotations

import unittest

from pipeline.depth import RelativeDepth


class NestingOrderTests(unittest.TestCase):
    """The order is the document's, not a table's."""

    def test_perpres_order_digits_then_letters(self) -> None:
        engine = RelativeDepth()
        self.assertEqual(engine.depth_for("decimal_plain", "1", 72.0), 0)
        self.assertEqual(engine.depth_for("decimal_dotted", "1.1", 100.0), 1)
        self.assertEqual(engine.depth_for("latin_lower", "a", 120.0), 2)
        self.assertEqual(engine.depth_for("paren_digit", "1", 140.0), 3)

    def test_inverted_order_letters_above_digits(self) -> None:
        """`A.` > `1.` > `a.` > `1)`: the order the fixed table inverted."""
        engine = RelativeDepth()
        self.assertEqual(engine.depth_for("letter_upper", "A", 72.0), 0)
        self.assertEqual(engine.depth_for("decimal_plain", "1", 90.0), 1)
        self.assertEqual(engine.depth_for("latin_lower", "a", 110.0), 2)
        self.assertEqual(engine.depth_for("paren_digit", "1", 130.0), 3)

    def test_roman_parts_above_letters(self) -> None:
        engine = RelativeDepth()
        self.assertEqual(engine.depth_for("roman_upper", "I", 72.0), 0)
        self.assertEqual(engine.depth_for("letter_upper", "A", 90.0), 1)
        self.assertEqual(engine.depth_for("decimal_plain", "1", 110.0), 2)

    def test_siblings_reuse_their_level(self) -> None:
        engine = RelativeDepth()
        engine.depth_for("letter_upper", "A", 72.0)
        engine.depth_for("decimal_plain", "1", 90.0)
        engine.depth_for("latin_lower", "a", 110.0)
        self.assertEqual(engine.depth_for("decimal_plain", "2", 90.0), 1)   # closes "a."
        self.assertEqual(engine.depth_for("letter_upper", "B", 72.0), 0)    # closes "2."


class LabelledParentageTests(unittest.TestCase):
    def test_dotted_label_attaches_to_its_prefix(self) -> None:
        """"21.4" states its own parent, whatever the indent says."""
        engine = RelativeDepth()
        engine.depth_for("decimal_plain", "21", 72.0)
        self.assertEqual(engine.depth_for("decimal_dotted", "21.4", 300.0), 1)
        self.assertEqual(engine.depth_for("decimal_dotted", "21.5", 300.0), 1)

    def test_deeper_dotted_label(self) -> None:
        engine = RelativeDepth()
        engine.depth_for("decimal_plain", "33", 72.0)
        engine.depth_for("decimal_dotted", "33.8", 200.0)
        self.assertEqual(engine.depth_for("decimal_dotted", "33.8.1", 220.0), 2)


class RealDocumentHazardTests(unittest.TestCase):
    def test_centred_heading_still_takes_children(self) -> None:
        """A Surat Perjanjian centres "Pasal 5" (x0=316) while its ayat start at
        the left margin (x0=113). Popping on indent would orphan every ayat."""
        engine = RelativeDepth()
        self.assertEqual(engine.depth_for("article_word", "Pasal 5", 316.0), 0)
        self.assertEqual(engine.depth_for("paren_digit_both", "1", 113.0), 1)
        self.assertEqual(engine.depth_for("paren_digit_both", "2", 113.0), 1)

    def test_numbering_shaped_prose_does_not_open_a_level(self) -> None:
        """A body line beginning "38.2 dan 38.3 hanya berlaku ..." matches the
        dotted style. At the same indent as the level it sits in, it is a
        sibling, so what follows is not buried one step too deep."""
        engine = RelativeDepth()
        engine.depth_for("article_word", "Pasal 1", 72.0)
        stray = engine.depth_for("decimal_dotted", "38.2", 84.0)
        self.assertEqual(engine.depth_for("paren_digit_both", "1", 88.0), stray)

    def test_top_level_style_resets_everything(self) -> None:
        engine = RelativeDepth()
        engine.depth_for("letter_upper", "A", 72.0)
        engine.depth_for("decimal_plain", "1", 90.0)
        engine.depth_for("latin_lower", "a", 110.0)
        self.assertEqual(engine.depth_for("article_word", "Pasal 1", 300.0), 0)
        self.assertEqual(engine.depth_for("paren_digit_both", "1", 80.0), 1)

    def test_out_dent_closes_deeper_levels(self) -> None:
        engine = RelativeDepth()
        engine.depth_for("decimal_plain", "1", 72.0)
        engine.depth_for("latin_lower", "a", 120.0)
        engine.depth_for("paren_digit", "1", 150.0)
        self.assertEqual(engine.depth_for("bullet", "•", 118.0), 1)

    def test_reset_clears_state(self) -> None:
        engine = RelativeDepth()
        engine.depth_for("decimal_plain", "1", 72.0)
        engine.reset()
        self.assertEqual(engine.depth_for("latin_lower", "a", 200.0), 0)


if __name__ == "__main__":
    unittest.main()
