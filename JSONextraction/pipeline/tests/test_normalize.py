"""`pipeline.normalize` — Indonesian currency, dates, number words and rates."""
from __future__ import annotations

import unittest

from pipeline.normalize import parse_currency_id, parse_date_id, parse_number_words_id, parse_rate


class CurrencyTests(unittest.TestCase):
    def test_indonesian_format(self) -> None:
        self.assertEqual(parse_currency_id("Rp 1.500.000.000,00"), 1_500_000_000.00)
        self.assertEqual(parse_currency_id("Rp10.000"), 10_000)
        self.assertEqual(parse_currency_id("1.500.000.000,00"), 1_500_000_000.00)

    def test_dotted_rp_prefix_does_not_break_parsing(self) -> None:
        """Fixed 2026-09-21 (fix_plan Phase 1). Before: `[\\d.,]+` matched the
        abbreviation's own dot, stripped to "" and returned None, so a filled
        `Nilai Kontrak : Rp. 1.500.000.000,00` read as a template blank."""
        self.assertEqual(parse_currency_id("Rp. 1.500.000.000,00"), 1_500_000_000.00)
        self.assertEqual(parse_currency_id("Rp. 250.000,-"), 250_000)

    def test_unparseable_is_none(self) -> None:
        self.assertIsNone(parse_currency_id("Rp ........."))
        self.assertIsNone(parse_currency_id(""))

    @unittest.expectedFailure
    def test_anglo_format_is_not_misread(self) -> None:
        """`1,500,000.00` is read as 1.5 today. A mixed-currency contract needs
        the separator convention to be a locale setting (audit Tier 3)."""
        self.assertEqual(parse_currency_id("USD 1,500,000.00"), 1_500_000.00)


class DateTests(unittest.TestCase):
    def test_indonesian_textual_date(self) -> None:
        self.assertEqual(parse_date_id("12 Agustus 2024"), ("2024-08-12", "day"))
        self.assertEqual(parse_date_id("3 Des 2021"), ("2021-12-03", "day"))

    def test_numeric_date_is_day_first(self) -> None:
        self.assertEqual(parse_date_id("05/07/2024"), ("2024-07-05", "day"))

    def test_impossible_month_swaps_to_day_first(self) -> None:
        self.assertEqual(parse_date_id("13/05/2024"), ("2024-05-13", "day"))

    def test_fiscal_year_is_year_precision(self) -> None:
        self.assertEqual(parse_date_id("TAHUN ANGGARAN 2023"), ("2023", "year"))

    def test_bare_four_digit_number_is_not_a_date(self) -> None:
        """Fixed 2026-09-21 (fix_plan Phase 1). A year now needs a label; before,
        any 4-digit number qualified and `Pasal 1266` became a date."""
        self.assertEqual(parse_date_id("Pasal 1266 KUHPerdata"), (None, "none"))
        self.assertEqual(parse_date_id("Rp 2500"), (None, "none"))

    def test_labelled_year_still_resolves(self) -> None:
        self.assertEqual(parse_date_id("Nomor 2 Tahun 2017"), ("2017", "year"))

    @unittest.expectedFailure
    def test_english_month_names(self) -> None:
        self.assertEqual(parse_date_id("12 August 2024"), ("2024-08-12", "day"))


class NumberWordTests(unittest.TestCase):
    def test_compound_numbers(self) -> None:
        self.assertEqual(parse_number_words_id("seratus dua puluh"), 120)
        self.assertEqual(parse_number_words_id("dua ratus tujuh puluh"), 270)
        self.assertEqual(parse_number_words_id("sembilan puluh"), 90)

    def test_teens(self) -> None:
        self.assertEqual(parse_number_words_id("dua belas"), 12)
        self.assertEqual(parse_number_words_id("sebelas"), 11)

    def test_non_numeric_text_is_none(self) -> None:
        self.assertIsNone(parse_number_words_id("hari kalender"))


class RateTests(unittest.TestCase):
    def test_surface_forms_unify(self) -> None:
        self.assertAlmostEqual(parse_rate("1‰"), 0.001)
        self.assertAlmostEqual(parse_rate("1/1000"), 0.001)
        self.assertAlmostEqual(parse_rate("5%"), 0.05)

    def test_ocr_permille_is_checked_before_percent(self) -> None:
        """Tesseract renders ‰ as "%o"; read as a percent it is 1000x wrong."""
        self.assertAlmostEqual(parse_rate("1 %o dari nilai kontrak"), 0.001)

    def test_comma_decimal(self) -> None:
        self.assertAlmostEqual(parse_rate("0,5%"), 0.005)


if __name__ == "__main__":
    unittest.main()
