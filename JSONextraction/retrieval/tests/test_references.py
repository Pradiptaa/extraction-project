"""`retrieval.references` — reading a structural reference out of a question and
matching it against stored labels.

Label shapes here are invented, not copied from the specimens: the matcher must
work for any contract's numbering, not this corpus's.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import unittest

from retrieval.references import Citation, find, match_tier, merge, normalize_path, parse
from retrieval.retrievers import Hit


def _segments(question: str):
    citation = parse(question)
    return None if citation is None else citation.segments


class ParseTests(unittest.TestCase):
    def test_article_and_ayat_in_every_common_spelling(self) -> None:
        for question in ("Pasal 5 ayat (3)", "pasal 5 ayat 3", "PASAL 5 AYAT 3",
                         "psl. 5 ayat 3", "ps 5 ayat (3)", "Pasal 5 (3)", "pasal 5 ayat [3]"):
            with self.subTest(question=question):
                self.assertEqual(_segments(question), ("5", "3"))

    def test_dotted_clause_numbers(self) -> None:
        for question in ("pasal 33.8", "klausul 33.8", "angka 33.8", "butir 33.8", "clause 33.8"):
            with self.subTest(question=question):
                self.assertEqual(_segments(question), ("33.8",))

    def test_deeper_addresses(self) -> None:
        self.assertEqual(_segments("Pasal 39.2 huruf a"), ("39.2", "a"))
        self.assertEqual(_segments("pasal 4 ayat 2 huruf a"), ("4", "2", "a"))
        self.assertEqual(_segments("bab ix"), ("ix",))

    def test_a_part_name_can_stand_in_for_the_level_word(self) -> None:
        citation = parse("SSUK 33.8")
        self.assertEqual(citation.segments, ("33.8",))
        self.assertEqual(citation.part, "general_terms")

    def test_the_part_hint_is_read_from_anywhere_in_the_question(self) -> None:
        self.assertEqual(parse("klausul 33.8 SSKK").part, "special_terms")
        self.assertEqual(parse("apa isi pasal 5 ayat 3 Surat Perjanjian").part, "main_agreement")
        self.assertEqual(parse("lampiran a angka 3").part, "annex_a")
        self.assertIsNone(parse("pasal 33.8").part)

    def test_the_rest_of_the_question_survives_for_search(self) -> None:
        citation = parse("Berapa lama Masa Pemeliharaan sebagaimana tercantum dalam Pasal 5 ayat (3)")
        self.assertEqual(citation.remainder, "Berapa lama Masa Pemeliharaan")
        self.assertEqual(parse("apa isi pasal 33.8?").remainder, "apa")
        self.assertEqual(parse("Pasal 5 ayat 3").remainder, "")

    def test_numbers_that_are_not_references_are_ignored(self) -> None:
        """A level word is always required, so amounts and durations are safe."""
        for question in ("berapa denda keterlambatan?", "masa pemeliharaan 180 hari kalender",
                         "nilai kontrak Rp33.8 miliar", "kontrak tanggal 5 April 2023",
                         "Peraturan Presiden Nomor 16 Tahun 2018", "keadaan kahar"):
            with self.subTest(question=question):
                self.assertIsNone(parse(question))

    def test_a_reference_relative_to_an_unnamed_clause_is_ignored(self) -> None:
        self.assertIsNone(parse("sebagaimana dimaksud dalam huruf b"))

    def test_str_is_readable_for_command_output(self) -> None:
        self.assertEqual(str(parse("pasal 5 ayat (3)")), "Pasal 5 ayat 3")


class NormalizeTests(unittest.TestCase):
    def test_level_words_and_brackets_are_stripped(self) -> None:
        self.assertEqual(normalize_path("Pasal 5/3"), (["pasal", ""], ["5", "3"]))
        self.assertEqual(normalize_path("BAB IX/Pasal 12/(2)"), (["bab", "pasal", ""], ["ix", "12", "2"]))


class MatchTests(unittest.TestCase):
    def test_the_level_word_decides_between_rows_sharing_numbers(self) -> None:
        citation = parse("pasal 5 ayat 3")
        self.assertEqual(match_tier(citation, "Pasal 5/3"), 0)
        self.assertEqual(match_tier(citation, "5/3"), 1, "a list also numbered 5/3 matches, but lower")

    def test_a_different_article_with_the_cited_number_inside_is_not_a_match(self) -> None:
        self.assertIsNone(match_tier(parse("pasal 5 ayat 3"), "Pasal 3/5"))
        self.assertEqual(match_tier(parse("pasal 5"), "Pasal 2/5"), 1, "numbers only")
        self.assertEqual(match_tier(parse("pasal 5"), "Pasal 5"), 0)

    def test_organisational_prefixes_are_optional(self) -> None:
        """A reader cites 12.4, never the book or section it was filed under."""
        citation = parse("pasal 12.4")
        for path in ("12/12.4", "B/12/12.4", "BAB II/Bagian 3/12/12.4"):
            with self.subTest(path=path):
                self.assertIsNotNone(match_tier(citation, path))

    def test_split_and_joined_numbering_are_equivalent(self) -> None:
        self.assertIsNotNone(match_tier(parse("pasal 12.4"), "Pasal 12/4"))
        self.assertIsNotNone(match_tier(parse("pasal 12 ayat 4"), "12.4"))

    def test_numbering_the_source_does_not_use_never_matches(self) -> None:
        """Renumbered documents must miss honestly rather than match a neighbour."""
        self.assertIsNone(match_tier(parse("pasal 33.8"), "B/B.3/1.82"))
        self.assertIsNone(match_tier(parse("pasal 33.8"), "33/33.80"))
        self.assertIsNone(match_tier(parse("pasal 5 ayat 3"), "t_062_0/3"))

    def test_deeper_addresses_match_their_own_level(self) -> None:
        self.assertEqual(match_tier(parse("pasal 4 ayat 2 huruf a"), "Pasal 4/2/a"), 0)
        self.assertEqual(match_tier(parse("pasal 39.2 huruf a"), "39/39.2/a"), 1)
        self.assertIsNone(match_tier(parse("pasal 39.2 huruf a"), "39/39.2/b"))


class FakeCollection:
    """Enough of the Chroma read interface for `find`."""

    def __init__(self, rows) -> None:
        self.rows = rows

    def get(self, where=None, include=None):
        keep = None
        if where:
            keep = set(where["document_key"]["$in"])
        rows = [row for row in self.rows if keep is None or row[1]["document_key"] in keep]
        return {"ids": [r[0] for r in rows], "metadatas": [r[1] for r in rows],
                "documents": [r[2] for r in rows]}


def _row(row_id, path, text, document="doc_a", sub_document="main_agreement", node_type="subclause"):
    return (row_id, {"document_key": document, "hierarchy_path": path, "sub_document": sub_document,
                     "node_type": node_type, "label": path.rsplit("/", 1)[-1]}, text)


class FindTests(unittest.TestCase):
    ROWS = [
        _row("a1", "Pasal 5", "MASA KONTRAK", node_type="article"),
        _row("a2", "Pasal 5/1", "Masa Kontrak adalah jangka waktu berlakunya Kontrak ini sampai penyerahan akhir."),
        _row("a3", "Pasal 5/2", "Masa Pelaksanaan ditentukan dalam SSKK selama 120 hari kalender."),
        _row("a4", "5/3", "telah membaca dan memahami secara penuh ketentuan Kontrak ini;", node_type="list_item"),
        _row("a5", "B/33/33.8", "Masa Pemeliharaan paling singkat 6 bulan.", sub_document="general_terms"),
        _row("b1", "Pasal 5/2", "Masa Pelaksanaan ditentukan dalam SSKK selama 90 hari kalender.", document="doc_b"),
    ]

    def setUp(self) -> None:
        self.collection = FakeCollection(self.ROWS)

    def test_only_the_best_tier_is_returned(self) -> None:
        result = find(parse("pasal 5 ayat 1"), self.collection)
        self.assertEqual([hit.id for hit in result.hits], ["a2"])
        self.assertEqual(result.tier, 0)

    def test_a_lower_tier_is_used_when_nothing_better_exists(self) -> None:
        result = find(parse("pasal 5 ayat 3"), self.collection)
        self.assertEqual([hit.id for hit in result.hits], ["a4"])
        self.assertEqual(result.tier, 1)

    def test_scope_limits_which_documents_can_match(self) -> None:
        result = find(parse("pasal 5 ayat 2"), self.collection, scope={"doc_b"})
        self.assertEqual([hit.id for hit in result.hits], ["b1"])

    def test_without_scope_every_document_matches(self) -> None:
        result = find(parse("pasal 5 ayat 2"), self.collection)
        self.assertEqual({hit.id for hit in result.hits}, {"a3", "b1"})

    def test_a_named_part_narrows_the_match(self) -> None:
        result = find(parse("SSUK 33.8"), self.collection)
        self.assertEqual([hit.id for hit in result.hits], ["a5"])
        self.assertFalse(result.part_relaxed)

    def test_a_wrongly_filed_unit_is_still_found_and_flagged(self) -> None:
        """Extraction can misfile a unit; refusing it would hide what was asked for."""
        result = find(parse("SSKK 33.8"), self.collection)
        self.assertEqual([hit.id for hit in result.hits], ["a5"])
        self.assertTrue(result.part_relaxed)

    def test_a_cited_heading_is_expanded_to_its_children(self) -> None:
        result = find(parse("pasal 5"), self.collection, limit=5)
        self.assertEqual([hit.id for hit in result.hits], ["a1", "a2", "a3"])
        self.assertEqual(result.expanded, 2)

    def test_no_match_reports_nothing_rather_than_a_neighbour(self) -> None:
        result = find(parse("pasal 99 ayat 1"), self.collection)
        self.assertFalse(result.found)
        self.assertIsNone(result.tier)

    def test_the_limit_is_respected(self) -> None:
        result = find(parse("pasal 5 ayat 2"), self.collection, limit=1)
        self.assertEqual(len(result.hits), 1)


class MergeTests(unittest.TestCase):
    PINNED = [Hit(id="p", score=0.0, metadata={}, text="cited")]
    SEARCHED = [Hit(id="s1", score=1.0, metadata={}, text="one"),
                Hit(id="p", score=0.9, metadata={}, text="cited"),
                Hit(id="s2", score=0.8, metadata={}, text="two")]

    def test_cited_rows_come_first_and_search_fills_the_rest(self) -> None:
        self.assertEqual([hit.id for hit in merge(self.PINNED, self.SEARCHED, 3)], ["p", "s1", "s2"])

    def test_a_row_never_appears_twice(self) -> None:
        merged = merge(self.PINNED, self.SEARCHED, 5)
        self.assertEqual(len([hit for hit in merged if hit.id == "p"]), 1)

    def test_k_is_never_exceeded(self) -> None:
        self.assertEqual(len(merge(self.PINNED, self.SEARCHED, 2)), 2)

    def test_no_citation_leaves_the_search_result_untouched(self) -> None:
        self.assertEqual([hit.id for hit in merge([], self.SEARCHED, 3)], ["s1", "p", "s2"])


class CitationDataTests(unittest.TestCase):
    def test_a_citation_can_be_built_directly(self) -> None:
        citation = Citation(segments=("7", "2"), levels=("pasal", "ayat"), part=None, remainder="", raw="Pasal 7 ayat 2")
        self.assertEqual(match_tier(citation, "Pasal 7/2"), 0)


if __name__ == "__main__":
    unittest.main()
