"""`pipeline.field_context` and the segment-aware core fields (fix_plan Phase 6).

Every case here is a value the extractor used to report as settled while it was
wrong — the class that defeats the review gate, because a `null` asks for a
human and a confident wrong answer does not.
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
from pipeline.field_context import FieldContext
from vocabulary import for_profile_id

GOODS_TITLE = "SURAT PERJANJIAN\nPENGADAAN BARANG\nNomor Kontrak : 77/SP-BARANG/2024\n"
FRONT_MATTER = "lanjutan pembukaan perjanjian ini\n"
# Deep body text: names construction, and carries a label word mid-sentence.
SSUK_BODY = (
    "Penyedia melaksanakan pekerjaan konstruksi sesuai KONTRAK KERJA KONSTRUKSI yang berlaku.\n"
    "Persyaratan pekerjaan yang Subkontraktor laksanakan harus memperhatikan mutu.\n"
)


def context_for(pages: dict[int, str], sub_documents: dict[int, str | None] | None = None) -> FieldContext:
    order = sorted(pages)
    return FieldContext.from_pages(order, pages, sub_documents or {p: None for p in order})


class FieldContextTests(unittest.TestCase):
    def test_offsets_map_back_to_their_page(self) -> None:
        ctx = context_for({1: "halaman satu", 2: "halaman dua"})
        self.assertEqual(ctx.page_at(0), 1)
        self.assertEqual(ctx.page_at(ctx.full_text.index("dua")), 2)

    def test_full_text_matches_the_joined_pages(self) -> None:
        pages = {1: "satu", 2: "dua", 3: "tiga"}
        self.assertEqual(context_for(pages).full_text, "satu\n\ndua\n\ntiga")

    def test_sub_documents_are_reported_per_offset(self) -> None:
        ctx = context_for({1: "pembukaan", 2: "isi ssuk"}, {1: "main_agreement", 2: "general_terms"})
        self.assertEqual(ctx.sub_document_at(0), "main_agreement")
        self.assertEqual(ctx.sub_document_at(ctx.full_text.index("ssuk")), "general_terms")
        self.assertEqual(ctx.first_sub_document(), "main_agreement")

    def test_the_title_region_covers_the_opening_pages_only(self) -> None:
        ctx = context_for({1: GOODS_TITLE, 2: FRONT_MATTER, 3: SSUK_BODY})
        self.assertTrue(ctx.in_title_region(0))
        self.assertFalse(ctx.in_title_region(ctx.full_text.index("Penyedia melaksanakan")))


class DocumentTypeByTitleRegionTests(unittest.TestCase):
    PAGES = {1: GOODS_TITLE, 2: FRONT_MATTER, 3: SSUK_BODY}

    def test_a_goods_contract_is_not_typed_as_construction(self) -> None:
        """"SURAT PERJANJIAN" is said by every contract in the family, and the
        body mentions construction; the title says what this document is."""
        ctx = context_for(self.PAGES)
        self.assertEqual(resolve_document_type(ctx.full_text, None, ctx)["value"], "kontrak_pengadaan_barang")

    def test_without_a_context_the_old_reading_is_kept(self) -> None:
        """The legacy engine is frozen, so the same text resolves as before."""
        ctx = context_for(self.PAGES)
        self.assertEqual(resolve_document_type(ctx.full_text)["value"], "kontrak_konstruksi")


class LabelledFieldTests(unittest.TestCase):
    def test_a_label_inside_a_sentence_is_not_the_contract_name(self) -> None:
        ctx = context_for({1: "SURAT PERJANJIAN\n", 2: FRONT_MATTER, 3: SSUK_BODY})
        self.assertIsNone(resolve_contract_name(ctx.full_text, None, ctx)["value"])

    def test_a_title_block_label_still_resolves(self) -> None:
        ctx = context_for({
            1: "SURAT PERJANJIAN\nNama Pekerjaan : Rehabilitasi Gedung Kantor\n",
            2: FRONT_MATTER, 3: SSUK_BODY,
        })
        self.assertEqual(resolve_contract_name(ctx.full_text, None, ctx)["value"], "Rehabilitasi Gedung Kantor")

    def test_a_blank_template_name_is_unfilled_not_a_value(self) -> None:
        ctx = context_for({1: "Nama Pekerjaan : ........................ [diisi nama paket pekerjaan]\n"})
        got = resolve_contract_name(ctx.full_text, None, ctx)
        self.assertIsNone(got["value"])
        self.assertIn("template_placeholder", got["flags"])

    def test_contract_number_reads_from_the_title_block(self) -> None:
        ctx = context_for({1: GOODS_TITLE})
        self.assertEqual(resolve_contract_number(ctx.full_text, None, ctx)["value"], "77/SP-BARANG/2024")


class PartyTests(unittest.TestCase):
    # Two copies of the agreement, as a bound-in specimen form produces: the
    # first filled, the second blank.
    TWO_COPIES = (
        'Nama : ADI RUDINI, ST\nJabatan : Kepala Satuan Kerja\n'
        'bertindak untuk dan atas nama Dinas Tenaga Kerja, selanjutnya disebut "Pejabat Penandatangan";\n\n'
        'Nama : SITI AMINAH\nJabatan : Direktur\n'
        'bertindak untuk dan atas nama PT Karya Bangun, selanjutnya disebut "Penyedia";\n\n'
        'CONTOH 2\n'
        'Nama : ....................\nJabatan : ....................\n'
        'bertindak untuk dan atas nama ...................., selanjutnya disebut "Pejabat Penandatangan";\n\n'
        'Nama : ....................\nJabatan : ....................\n'
        'bertindak untuk dan atas nama ...................., selanjutnya disebut "Penyedia";\n'
    )

    def test_a_repeated_blank_copy_does_not_double_the_parties(self) -> None:
        """A specimen form bound into the contract repeats the party block, and
        two real parties came out as four."""
        ctx = context_for({1: self.TWO_COPIES})
        parties = resolve_parties(ctx.full_text, None, ctx)["value"]
        self.assertEqual(len(parties), 2)
        self.assertEqual([p["representative"]["name"] for p in parties], ["ADI RUDINI, ST", "SITI AMINAH"])

    def test_without_a_context_the_repeats_are_kept(self) -> None:
        """Frozen behaviour: the legacy engine still reports all four."""
        self.assertEqual(len(resolve_parties(self.TWO_COPIES)["value"]), 4)

    def test_the_filled_copy_wins_over_the_blank_one(self) -> None:
        reversed_order = "\n".join(self.TWO_COPIES.split("CONTOH 2")[::-1])
        ctx = context_for({1: reversed_order})
        names = [p["representative"]["name"] for p in resolve_parties(ctx.full_text, None, ctx)["value"]]
        self.assertIn("ADI RUDINI, ST", names)


class DateConfidenceTests(unittest.TestCase):
    UNLABELLED = "Berlaku 12 Agustus 2024 sampai 30 Desember 2024 sesuai ketentuan.\n"

    def test_dates_that_cannot_be_told_apart_stay_below_review(self) -> None:
        """They parsed, but which one is the signing date is unknown — reporting
        that at 1.0 hides a recall failure behind a parsing success."""
        ctx = context_for({1: self.UNLABELLED})
        got = resolve_key_dates(ctx.full_text, None, ctx)
        self.assertLess(got["confidence"], 0.6)
        self.assertIn("dates_unclassified", got["flags"])

    def test_a_classified_date_keeps_its_confidence(self) -> None:
        ctx = context_for({1: "Ditetapkan di Sukamara pada tanggal 12 Agustus 2024\n"})
        got = resolve_key_dates(ctx.full_text, None, ctx)
        self.assertGreaterEqual(got["confidence"], 0.6)

    def test_without_a_context_the_confidence_is_unchanged(self) -> None:
        self.assertGreaterEqual(resolve_key_dates(self.UNLABELLED)["confidence"], 0.6)


class DurationSourceTests(unittest.TestCase):
    PAGES = {
        1: "Masa Pemeliharaan selama 180 (seratus delapan puluh) hari kalender.\n",
        2: "Masa Pemeliharaan paling singkat untuk pekerjaan permanen selama 6 (enam) bulan.\n",
    }

    def _durations(self, sub_documents, vocab=None) -> list[dict]:
        ctx = context_for(self.PAGES, sub_documents)
        got = resolve_key_numbers(ctx.full_text, vocab or for_profile_id("perpres16_konstruksi_v1"), ctx)
        return [n for n in got["value"] if n["type"] == "duration"]

    def test_values_in_a_standard_form_part_are_marked_as_such(self) -> None:
        got = {(n["amount"], n["unit"]): n for n in self._durations({1: "main_agreement", 2: "general_terms"})}
        self.assertEqual(got[(180, "hari_kalender")]["source"], "contract")
        self.assertEqual(got[(6, "bulan")]["source"], "standard_form")
        self.assertEqual(got[(6, "bulan")]["sub_document"], "general_terms")

    def test_a_value_repeated_in_the_contract_part_counts_as_the_contract_s(self) -> None:
        self.PAGES = {1: self.PAGES[2], 2: self.PAGES[2]}
        got = self._durations({1: "general_terms", 2: "special_terms"})
        self.assertEqual([(n["amount"], n["source"]) for n in got], [(6, "contract")])

    def test_without_a_context_no_source_is_claimed(self) -> None:
        got = resolve_key_numbers("".join(self.PAGES.values()), for_profile_id("perpres16_konstruksi_v1"))
        self.assertTrue(all("source" not in n for n in got["value"] if n["type"] == "duration"))


if __name__ == "__main__":
    unittest.main()
