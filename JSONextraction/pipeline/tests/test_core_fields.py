"""`pipeline.core_fields` — the regex/heuristic cascade behind the six core fields.

Short synthetic documents, not specimens: these pin the *rules*, while the
specimens pin the outputs. `expectedFailure` marks a known generalization gap
from md/fix_plan.md.
"""
from __future__ import annotations

import unittest

from pipeline.core_fields import (
    resolve_contract_name,
    resolve_contract_number,
    resolve_document_type,
    resolve_key_dates,
    resolve_key_numbers,
    resolve_parties,
)

FILLED_PARTIES = """
SURAT PERJANJIAN
Nomor : 027/SP/PUPR/2024

Nama : BUDI SANTOSO, S.T.
Jabatan : Pejabat Pembuat Komitmen
Berkedudukan di : Jl. Merdeka 1, Sukamara
NIP. 19810618 200604 2 008
bertindak untuk dan atas nama Dinas Pekerjaan Umum Kabupaten Sukamara,
selanjutnya disebut "PPK";

Nama : SITI AMINAH
Jabatan : Direktur
Berkedudukan di : Jl. Industri 9
bertindak untuk dan atas nama PT Karya Bangun Sejahtera, selanjutnya disebut "Penyedia";
"""


class DocumentTypeTests(unittest.TestCase):
    def test_construction_contract(self) -> None:
        got = resolve_document_type("SURAT PERJANJIAN KONTRAK KERJA KONSTRUKSI")
        self.assertEqual(got["value"], "kontrak_konstruksi")

    def test_subtype_is_detected(self) -> None:
        got = resolve_document_type("SURAT PERJANJIAN\nKONTRAK HARGA SATUAN")
        self.assertEqual(got["subtype"], "kontrak_harga_satuan")

    def test_no_signal_is_unknown_not_a_guess(self) -> None:
        got = resolve_document_type("NOTULEN RAPAT KOORDINASI")
        self.assertEqual(got["value"], "unknown")
        self.assertEqual(got["confidence"], 0.0)

    @unittest.expectedFailure
    def test_goods_contract_is_not_typed_as_construction(self) -> None:
        """"Surat Perjanjian" is a konstruksi signal, and every Perpres contract
        says it. fix_plan Phase 6 scores title-region matches instead."""
        got = resolve_document_type("SURAT PERJANJIAN\nPENGADAAN BARANG\nNomor: 1/PB/2024")
        self.assertEqual(got["value"], "kontrak_pengadaan_barang")


class ContractNumberTests(unittest.TestCase):
    def test_labeled_number(self) -> None:
        got = resolve_contract_number("Nomor Kontrak : 027/SP/PUPR/2024\n")
        self.assertEqual(got["value"], "027/SP/PUPR/2024")

    def test_legal_citation_is_not_the_contract_number(self) -> None:
        """"Nomor : X tentang Y" is a citation — a robust Indonesian legal rule."""
        got = resolve_contract_number("Peraturan Presiden Nomor : 16/2018 tentang Pengadaan\n")
        self.assertIsNone(got["value"])

    def test_blank_template_number_resolves_to_null(self) -> None:
        """An all-dots value never matches the value shape, so it is simply
        unresolved. A *partly* filled one is what carries the placeholder flag."""
        got = resolve_contract_number("Nomor : ......../SP/2024\n")
        self.assertIsNone(got["value"])
        self.assertIn("review_required", got["flags"])

    @unittest.expectedFailure
    def test_number_with_spaces(self) -> None:
        """The value shape forbids spaces, so this common form is missed."""
        got = resolve_contract_number("Nomor : 12 / PPK / 2024\n")
        self.assertEqual(got["value"], "12 / PPK / 2024")


class ContractNameTests(unittest.TestCase):
    def test_labeled_name(self) -> None:
        got = resolve_contract_name("Nama Pekerjaan : Rehabilitasi Gedung Kantor\n")
        self.assertEqual(got["value"], "Rehabilitasi Gedung Kantor")


class PartyTests(unittest.TestCase):
    def test_two_parties_from_disebut_definitions(self) -> None:
        got = resolve_parties(FILLED_PARTIES)
        self.assertEqual(len(got["value"]), 2)
        self.assertEqual([p["role_label"] for p in got["value"]], ["PPK", "Penyedia"])

    def test_representative_and_nip(self) -> None:
        ppk = resolve_parties(FILLED_PARTIES)["value"][0]
        self.assertEqual(ppk["representative"]["name"], "BUDI SANTOSO, S.T.")
        self.assertEqual(ppk["representative"]["identifier"]["value"], "19810618 200604 2 008")

    def test_self_referential_terms_are_not_parties(self) -> None:
        text = 'Perjanjian ini selanjutnya disebut "Kontrak";\n' + FILLED_PARTIES
        self.assertEqual(len(resolve_parties(text)["value"]), 2)

    def test_no_parties_is_flagged_not_invented(self) -> None:
        got = resolve_parties("Dokumen ini tidak memuat para pihak.")
        self.assertEqual(got["value"], [])
        self.assertIn("no_parties_detected", got["flags"])

    def test_disebut_path_reads_a_party_without_a_nip(self) -> None:
        """The `selanjutnya disebut` path reads `Nama :` labels, so a private
        counterparty with no NIP still resolves. Only the marker path below
        depends on a NIP — narrower than the audit's "parties need a NIP"."""
        text = (
            "Nama : SITI AMINAH\nJabatan : Direktur\n"
            "bertindak atas nama PT Karya Bangun, selanjutnya disebut \"Penyedia\";\n"
        )
        self.assertEqual(resolve_parties(text)["value"][0]["representative"]["name"], "SITI AMINAH")

    def test_marker_path_reads_names_without_a_nip(self) -> None:
        """Fixed 2026-09-21 (fix_plan Phase 6). The block's own `Nama :` label is
        read first; the "line before a NIP" rule is now only a fallback, so a
        contract between two private companies keeps both names."""
        text = (
            "PIHAK PERTAMA\nNama : BUDI SANTOSO\nJabatan : Direktur PT Karya Bangun\n\n"
            "PIHAK KEDUA\nNama : SITI AMINAH\nJabatan : Direktur PT Mitra Jaya\n"
        )
        got = resolve_parties(text)["value"]
        self.assertEqual([p["representative"]["name"] for p in got], ["BUDI SANTOSO", "SITI AMINAH"])
        self.assertEqual([p["flags"] for p in got], [[], []])
        self.assertEqual(got[0]["representative"]["position"], "Direktur PT Karya Bangun")


class KeyNumberTests(unittest.TestCase):
    def test_duration_with_words_check(self) -> None:
        got = resolve_key_numbers("Masa Pelaksanaan 180 (seratus delapan puluh) hari kalender")
        duration = next(n for n in got["value"] if n["type"] == "duration")
        self.assertEqual((duration["amount"], duration["subtype"]), (180, "masa_pelaksanaan"))
        self.assertEqual(duration["words_check"], "passed")

    def test_words_digits_mismatch_is_reported(self) -> None:
        got = resolve_key_numbers("Masa Pemeliharaan 180 (seratus) hari kalender")
        self.assertEqual(next(n for n in got["value"] if n["type"] == "duration")["words_check"], "mismatch")

    def test_penalty_rate_subtype(self) -> None:
        got = resolve_key_numbers("denda keterlambatan sebesar 1‰ dari nilai kontrak")
        penalty = next(n for n in got["value"] if n["type"] == "penalty_rate")
        self.assertEqual(penalty["subtype"], "denda_keterlambatan")
        self.assertAlmostEqual(penalty["amount"], 0.001)

    def test_meterai_is_not_promoted_to_contract_value(self) -> None:
        got = resolve_key_numbers("dibubuhi meterai Rp10.000")
        self.assertNotIn("contract_value", [n["type"] for n in got["value"]])

    @unittest.expectedFailure
    def test_value_label_does_not_match_a_heading(self) -> None:
        """BUG (found 2026-09-21): "Harga Kontrak" inside the heading "HARGA
        KONTRAK, SUMBER PEMBIAYAAN DAN PEMBAYARAN" is taken as the value, so
        three specimens carry `raw=", SUMBER PEMBIAYAAN DAN PEMBAYARAN"` as
        their contract value. fix_plan Phase 6 (segment-aware resolution)."""
        got = resolve_key_numbers("Pasal 3\nHARGA KONTRAK, SUMBER PEMBIAYAAN DAN PEMBAYARAN\n")
        self.assertEqual([n for n in got["value"] if n["type"] == "contract_value"], [])

    def test_duration_units_beyond_hari_kalender(self) -> None:
        """Fixed 2026-09-21 (fix_plan Phase 1). The unit is kept in the output
        rather than assumed to be `hari_kalender`."""
        for text, amount, unit in (
            ("Masa Pelaksanaan 90 (sembilan puluh) hari kerja", 90, "hari_kerja"),
            ("Masa Pemeliharaan 6 (enam) bulan", 6, "bulan"),
            ("Masa Pelaksanaan 180 (seratus delapan puluh) hari kalender", 180, "hari_kalender"),
        ):
            duration = next(n for n in resolve_key_numbers(text)["value"] if n["type"] == "duration")
            self.assertEqual((duration["amount"], duration["unit"]), (amount, unit), text)

    def test_bare_duration_without_words_needs_a_subtype(self) -> None:
        """Digits-only durations resolve when something names them, and are
        dropped in loose prose, where "30 hari" is usually not a contract term."""
        named = resolve_key_numbers("Masa Pemeliharaan 180 hari kalender")["value"]
        self.assertEqual(next(n for n in named if n["type"] == "duration")["words_check"], "no_words")
        loose = resolve_key_numbers("pekerjaan dilaksanakan dalam 30 hari")["value"]
        self.assertEqual([n for n in loose if n["type"] == "duration"], [])


class KeyDateTests(unittest.TestCase):
    def test_textual_date_is_resolved(self) -> None:
        got = resolve_key_dates("Ditetapkan di Sukamara pada tanggal 12 Agustus 2024")
        self.assertIn("2024-08-12", [d["date"] for d in got["value"]])

    def test_clause_citation_is_not_a_date(self) -> None:
        """Fixed 2026-09-21 (fix_plan Phase 1). The scan matched "66 dan 1267"
        as day/month/year, and the 4-digit fallback then dated it to 1267."""
        got = resolve_key_dates("mengesampingkan Pasal 1266 dan 1267 Kitab Undang-Undang Hukum Perdata")
        self.assertEqual(got["value"], [])

    def test_real_dates_still_scan(self) -> None:
        got = resolve_key_dates("pada tanggal 12 Agustus 2024 dan 05/07/2024, TAHUN ANGGARAN 2023")
        self.assertEqual([e["date"] for e in got["value"]], ["2024-08-12", "2024-07-05", "2023"])


if __name__ == "__main__":
    unittest.main()
