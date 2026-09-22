"""`pipeline.headings` and `pipeline.segments` (fix_plan Phase 4).

The cases here are the ones that cost real debugging: a capitalised sentence
inside a clause, a bold label line in a letterhead, a contents page that names
every part before any of them starts.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from pipeline.headings import PageStats, score_heading
from pipeline.segments import assign_sub_documents_by_block, find_toc_blocks, is_toc_line, page_level_view

STATS = PageStats(width=595.0, line_height=12.0, median_font_size=12.0, body_x0=72.0)


def score(text: str, *, x0=72.0, x1=520.0, bold=False, size=12.0, gap=30.0, previous="Kalimat sebelumnya."):
    return score_heading(text, stats=STATS, x0=x0, x1=x1, font_size=size, is_bold=bold,
                         gap_above=gap, previous_text=previous)


@dataclass
class FakeBlock:
    text: str
    x0: float = 72.0
    x1: float = 520.0
    top: float = 100.0
    bottom: float = 112.0
    font_size: float = 12.0
    is_bold: bool = False


class HeadingScoringTests(unittest.TestCase):
    def test_a_real_heading_scores_as_one(self) -> None:
        self.assertTrue(score("SYARAT-SYARAT UMUM KONTRAK").is_heading)
        self.assertTrue(score("SURAT PENUNJUKAN PENYEDIA BARANG/JASA (SPPBJ)").is_heading)

    def test_capitalised_sentence_in_a_body_is_not_a_heading(self) -> None:
        """The line that used to reset the whole ancestor stack."""
        self.assertFalse(score(
            "SELURUH KETENTUAN DALAM PASAL INI MENGIKAT PARA PIHAK SEJAK TANGGAL "
            "PENANDATANGANAN KONTRAK INI.").is_heading)

    def test_bold_label_line_is_not_a_heading(self) -> None:
        """"Perihal : ..." is a field, and letterheads set it in bold."""
        self.assertFalse(score("Perihal : Penunjukan Penyedia untuk Pelaksanaan Paket", bold=True).is_heading)
        self.assertFalse(score("Nomor : 027/SP/PUPR/2024", bold=True).is_heading)

    def test_short_isolated_prose_needs_emphasis(self) -> None:
        """Letters are full of short isolated lines that are not headings."""
        self.assertFalse(score("Pejabat Pembuat Komitmen Bidang Bina Marga").is_heading)

    def test_wrapped_body_line_is_not_centred_by_accident(self) -> None:
        wide = score("Kegagalan Bangunan adalah suatu keadaan keruntuhan", x0=85.0, x1=553.0)
        self.assertNotIn("centred", wide.signals)

    def test_line_continuing_a_sentence_is_body(self) -> None:
        self.assertFalse(score("dan seterusnya sebagaimana dimaksud pada ayat,").is_heading)

    def test_division_threshold_is_stricter(self) -> None:
        plain = score("KETENTUAN UMUM", gap=2.0, previous="teks berjalan")
        self.assertFalse(plain.is_division)
        emphasised = score("KETENTUAN UMUM", x0=200.0, x1=395.0, bold=True, size=16.0)
        self.assertTrue(emphasised.is_division)


class TableOfContentsTests(unittest.TestCase):
    def test_contents_lines_are_recognised(self) -> None:
        self.assertTrue(is_toc_line("SYARAT-SYARAT UMUM KONTRAK .................. 12"))
        self.assertTrue(is_toc_line("Pasal 5 Masa Kontrak     7"))
        self.assertFalse(is_toc_line("SYARAT-SYARAT UMUM KONTRAK"))

    def test_a_run_of_entries_is_found_without_taking_the_page(self) -> None:
        blocks = [
            FakeBlock("DAFTAR ISI"),
            FakeBlock("SURAT PERJANJIAN .................. 2"),
            FakeBlock("SYARAT-SYARAT UMUM KONTRAK .................. 3"),
            FakeBlock("SYARAT-SYARAT KHUSUS KONTRAK .................. 4"),
            FakeBlock("SURAT PERJANJIAN"),          # the real heading, same page
        ]
        toc = find_toc_blocks({1: blocks}, [1])
        self.assertEqual(toc, {(1, 0), (1, 1), (1, 2), (1, 3)})
        self.assertNotIn((1, 4), toc)


class SegmentationTests(unittest.TestCase):
    PROFILE = {
        "sub_document_markers": [
            {"name": "main_agreement", "start": "^SURAT PERJANJIAN"},
            {"name": "general_terms", "start": "^SYARAT-SYARAT UMUM KONTRAK"},
            {"name": "special_terms", "start": "^SYARAT-SYARAT KHUSUS KONTRAK"},
        ]
    }

    def blocks(self):
        return [
            FakeBlock("DAFTAR ISI", top=60, bottom=72),
            FakeBlock("SURAT PERJANJIAN .................. 2", top=90, bottom=102),
            FakeBlock("SYARAT-SYARAT UMUM KONTRAK .................. 3", top=110, bottom=122),
            FakeBlock("SYARAT-SYARAT KHUSUS KONTRAK .................. 4", top=130, bottom=142),
            FakeBlock("SURAT PERJANJIAN", top=200, bottom=212),
            FakeBlock("Isi perjanjian ini disepakati para pihak.", top=230, bottom=242),
            FakeBlock("SYARAT-SYARAT UMUM KONTRAK", top=300, bottom=312),
            FakeBlock("Ketentuan umum berlaku untuk seluruh pekerjaan.", top=330, bottom=342),
        ]

    def test_contents_page_does_not_start_a_part(self) -> None:
        assignment, notes = assign_sub_documents_by_block({1: self.blocks()}, [1], self.PROFILE, {1: 595.0})
        self.assertIsNone(assignment[(1, 1)])            # a contents entry
        self.assertEqual(assignment[(1, 4)], "main_agreement")
        self.assertEqual(assignment[(1, 7)], "general_terms")
        self.assertTrue(any(n.startswith("toc_blocks_skipped") for n in notes))

    def test_a_part_may_start_mid_page(self) -> None:
        assignment, _ = assign_sub_documents_by_block({1: self.blocks()}, [1], self.PROFILE, {1: 595.0})
        self.assertEqual(assignment[(1, 5)], "main_agreement")
        self.assertEqual(assignment[(1, 6)], "general_terms")

    def test_a_marker_quoted_in_prose_does_not_start_a_part(self) -> None:
        blocks = [
            FakeBlock("SURAT PERJANJIAN", top=60, bottom=72),
            FakeBlock("Ketentuan ini tunduk pada ketentuan berikut, yaitu:", top=90, bottom=102),
            FakeBlock("SYARAT-SYARAT UMUM KONTRAK yang berlaku bagi para pihak dan seluruh "
                      "pekerjaan yang diatur di dalamnya,", top=104, bottom=116),
        ]
        assignment, _ = assign_sub_documents_by_block({1: blocks}, [1], self.PROFILE, {1: 595.0})
        self.assertEqual(assignment[(1, 2)], "main_agreement")

    def test_missing_markers_are_reported(self) -> None:
        blocks = [FakeBlock("SURAT PERJANJIAN", top=60, bottom=72)]
        _, notes = assign_sub_documents_by_block({1: blocks}, [1], self.PROFILE, {1: 595.0})
        self.assertTrue(any("markers_not_found" in n for n in notes))

    def test_no_markers_means_no_assignment(self) -> None:
        assignment, notes = assign_sub_documents_by_block({1: self.blocks()}, [1], {}, {1: 595.0})
        self.assertEqual(assignment, {})
        self.assertEqual(notes, [])

    def test_page_view_reports_the_part_in_effect_at_the_page_end(self) -> None:
        """Not a majority vote: a page where a new part begins belongs to that
        part, because it owns the rest of the page and everything after. The
        vote put the SSKK's own page in general_terms — most of its surviving
        blocks belonged to the part that was ending — and every table row on
        that page inherited the wrong part."""
        assignment = {(1, 0): None, (1, 1): "main_agreement", (1, 2): "main_agreement", (1, 3): "general_terms"}
        self.assertEqual(page_level_view(assignment, [1]), {1: "general_terms"})

    def test_a_page_with_no_assignment_inherits_the_current_part(self) -> None:
        """A ruled-table page keeps only the blocks outside its tables, often
        none at all; it still belongs to the part it sits in."""
        assignment = {(1, 0): "general_terms"}
        self.assertEqual(page_level_view(assignment, [1, 2]), {1: "general_terms", 2: "general_terms"})


if __name__ == "__main__":
    unittest.main()
