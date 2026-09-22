"""`pipeline.numbering` — which numbering tokens are recognized, and as what.

Cases marked `expectedFailure` are the generalization gaps in md/fix_plan.md;
each should start passing when its phase lands, which makes the test a progress
marker rather than a wish.
"""
from __future__ import annotations

import unittest

from pipeline.numbering import match_numbering, sibling_successor


class RecognizedStylesTests(unittest.TestCase):
    def assert_style(self, text: str, style: str, label: str, remainder: str | None = None) -> None:
        m = match_numbering(text)
        self.assertIsNotNone(m, f"no numbering matched in {text!r}")
        self.assertEqual((m.style, m.label), (style, label))
        if remainder is not None:
            self.assertEqual(m.remainder, remainder)

    def test_article_and_chapter_words(self) -> None:
        self.assert_style("Pasal 5 MASA KONTRAK", "article_word", "Pasal 5", "MASA KONTRAK")
        self.assert_style("BAB II KETENTUAN UMUM", "chapter_word", "BAB II", "KETENTUAN UMUM")

    def test_decimal_styles(self) -> None:
        self.assert_style("1. Definisi", "decimal_plain", "1", "Definisi")
        self.assert_style("21.4 Pembayaran dilakukan", "decimal_dotted", "21.4", "Pembayaran dilakukan")
        self.assertEqual(match_numbering("21.4 x").depth_hint, 1)
        self.assertEqual(match_numbering("21.4.1 x").depth_hint, 2)

    def test_ayat_is_distinct_from_flat_list(self) -> None:
        """`(2)` and `2)` are different levels; conflating them merged a whole
        Pasal into one node (see ARCHITECTURE.md, numbering.py)."""
        self.assert_style("(2) Masa Pelaksanaan", "paren_digit_both", "2")
        self.assert_style("2) Masa Pelaksanaan", "paren_digit", "2")

    def test_letter_and_bullet_styles(self) -> None:
        self.assert_style("a. ketentuan umum", "latin_lower", "a")
        self.assert_style("A. KETENTUAN UMUM", "letter_upper", "A")
        self.assert_style("A.1 KETENTUAN", "letter_dotted", "A.1")
        self.assert_style("• ketentuan", "bullet", "•")

    def test_lowercase_pasal_is_not_a_heading(self) -> None:
        """An inline cross-reference ("...sebagaimana pasal 44.2") must not open
        a node; only the printed heading form does."""
        self.assertIsNone(match_numbering("pasal 44.2 tidak berlaku"))

    def test_civil_code_citation_is_not_an_article(self) -> None:
        """`\\d{1,3}` is what keeps "Pasal 1266 KUHPerdata" out of the tree; two
        specimens' ground truth documents this exact false positive."""
        self.assertIsNone(match_numbering("Pasal 1266 dan 1267 Kitab Undang-Undang"))

    def test_successor_rules(self) -> None:
        self.assertEqual(sibling_successor("decimal_dotted", "21.4"), "21.5")
        self.assertEqual(sibling_successor("latin_lower", "a"), "b")
        self.assertEqual(sibling_successor("article_word", "Pasal 5"), "Pasal 6")
        self.assertIsNone(sibling_successor("bullet", "•"))


class RomanSeriesTests(unittest.TestCase):
    """"I." is both a section letter and a roman numeral. Which one it is
    depends on the series it sits in, which is why it is resolved per document
    rather than per line."""

    def resolve(self, lines: list[str]) -> list[str]:
        from pipeline.numbering import resolve_roman_series

        return [m.style if m else "" for m in resolve_roman_series([match_numbering(t) for t in lines])]

    def test_lone_i_joins_its_roman_series(self) -> None:
        styles = self.resolve(["I. KETENTUAN UMUM", "A. Definisi", "1. Satu", "II. PELAKSANAAN"])
        self.assertEqual(styles[0], "roman_upper")
        self.assertEqual(styles[1], "letter_upper")

    def test_series_members_may_be_far_apart(self) -> None:
        """"I." and "II." are separated by the whole of part I, so neighbouring
        markers never see each other."""
        lines = ["I. BAGIAN SATU"] + [f"{n}. Butir" for n in range(1, 12)] + ["II. BAGIAN DUA"]
        self.assertEqual(self.resolve(lines)[0], "roman_upper")

    def test_alphabetic_run_keeps_its_letters(self) -> None:
        self.assertEqual(self.resolve(["G. GARANSI", "H. HAK", "I. INFORMASI", "J. JAMINAN"]),
                         ["letter_upper"] * 4)

    def test_roman_successor(self) -> None:
        self.assertEqual(sibling_successor("roman_upper", "IV"), "V")
        self.assertEqual(sibling_successor("roman_upper", "VIII"), "IX")


class UnsupportedStylesTests(unittest.TestCase):
    """Nesting orders and heading forms found outside the Perpres-16 family.
    fix_plan.md Phase 3 (relative depth) and the numbering rows of the audit."""

    def test_uppercase_roman_section(self) -> None:
        """Fixed 2026-09-21 (fix_plan Phase 3). Multi-character romans are
        unambiguous; a lone "I." is decided by its series below."""
        self.assertEqual(match_numbering("II. PELAKSANAAN").style, "roman_upper")
        self.assertEqual(match_numbering("IV. PENUTUP").style, "roman_upper")

    @unittest.expectedFailure
    def test_paragraf_and_ayat_heading_words(self) -> None:
        self.assertIsNotNone(match_numbering("Paragraf 2 Pembayaran"))

    @unittest.expectedFailure
    def test_lowercase_heading_after_letter(self) -> None:
        """`letter_upper` requires an uppercase letter next, so a sentence-case
        section title is missed."""
        self.assertIsNotNone(match_numbering("A. pembayaran prestasi pekerjaan"))

    @unittest.expectedFailure
    def test_article_with_letter_suffix(self) -> None:
        m = match_numbering("Pasal 5A KETENTUAN TAMBAHAN")
        self.assertEqual(m.label, "Pasal 5A")


if __name__ == "__main__":
    unittest.main()
