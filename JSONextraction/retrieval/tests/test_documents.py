"""Reading the contract a question names, so `--document` becomes optional.

Two risks are pinned here rather than left to judgement: scoping to a contract
the question did not name, and stopping an ordinary question because a common
word looked like a filename. The gate's own 20 queries are replayed against the
parser for the second.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from retrieval.documents import normalize, parse

CORPUS = {
    "k1": "Rancangan Kontrak.pdf",
    "k2": "kontrakJasa.pdf",
    "k3": "pembangunanRumah.pdf",
    "k4": "pembangunanSayap.pdf",
    "k5": "polres.pdf",
    "k6": "rehabGedung.pdf",
}

QUERY_SET = Path(__file__).resolve().parents[2] / "ground_truth" / "retrieval_queries.json"


class NormalizeTests(unittest.TestCase):
    def test_spacing_and_case_do_not_matter(self) -> None:
        """Filenames are camelCase; people type words with spaces."""
        self.assertEqual(normalize("pembangunan rumah"), normalize("pembangunanRumah"))

    def test_punctuation_is_dropped(self) -> None:
        self.assertEqual(normalize("Rancangan-Kontrak_v2"), "rancangankontrakv2")


class RecognisedTests(unittest.TestCase):
    def _scope(self, question: str) -> str:
        mention = parse(question, CORPUS)
        self.assertTrue(mention.found, f"expected a document in {question!r}: {mention.problem}")
        return next(iter(mention.scope.values()))

    def test_a_named_file_is_recognised(self) -> None:
        self.assertEqual(
            self._scope("Pada file Rancangan Kontrak, siapa saja pihak yang terlibat?"),
            "Rancangan Kontrak.pdf",
        )

    def test_dokumen_is_a_cue_too(self) -> None:
        self.assertEqual(self._scope("di dokumen rehabGedung berapa masa pemeliharaan?"),
                         "rehabGedung.pdf")

    def test_a_spaced_name_matches_a_camel_case_file(self) -> None:
        self.assertEqual(self._scope("dalam file pembangunan rumah, apa kewajiban asuransi?"),
                         "pembangunanRumah.pdf")

    def test_a_bare_filename_is_recognised_anywhere(self) -> None:
        self.assertEqual(self._scope("rehabGedung.pdf berapa lama masa pemeliharaan?"),
                         "rehabGedung.pdf")

    def test_a_cue_before_a_filename_is_not_taken_for_part_of_it(self) -> None:
        self.assertEqual(self._scope("file pembangunanSayap.pdf: siapa pengawas pekerjaan?"),
                         "pembangunanSayap.pdf")

    def test_the_longer_of_two_matching_names_wins(self) -> None:
        """"pembangunan" alone matches two specimens; the second word decides."""
        self.assertEqual(self._scope("pada file pembangunan sayap, apa ruang lingkupnya?"),
                         "pembangunanSayap.pdf")


class RemainderTests(unittest.TestCase):
    """The mention is stripped before retrieval: "pada file Rancangan Kontrak"
    is words every contract contains, so leaving them in would rank the whole
    corpus on the strength of the scope."""

    def test_the_question_survives_the_strip(self) -> None:
        self.assertEqual(
            parse("Pada file Rancangan Kontrak, siapa saja pihak yang terlibat?", CORPUS).remainder,
            "siapa saja pihak yang terlibat?",
        )

    def test_a_name_running_into_the_question_takes_only_the_name(self) -> None:
        """No punctuation separates them, so the cut is by word, not to the end."""
        self.assertEqual(
            parse("di dokumen rehabGedung berapa masa pemeliharaan?", CORPUS).remainder,
            "berapa masa pemeliharaan?",
        )

    def test_an_extension_is_stripped_with_the_name(self) -> None:
        self.assertEqual(
            parse("rehabGedung.pdf berapa lama masa pemeliharaan?", CORPUS).remainder,
            "berapa lama masa pemeliharaan?",
        )


class RefusedTests(unittest.TestCase):
    def test_an_ambiguous_name_is_refused_with_the_candidates(self) -> None:
        mention = parse("Pada file kontrak, siapa para pihak?", CORPUS)
        self.assertFalse(mention.found)
        self.assertIn("matches 2 documents", mention.problem)
        self.assertIn("Rancangan Kontrak.pdf", mention.problem)
        self.assertIn("kontrakJasa.pdf", mention.problem)

    def test_a_named_file_that_does_not_exist_is_reported(self) -> None:
        """`file` is never written by accident, so a miss is the user's answer,
        not a reason to quietly search all six."""
        mention = parse("pada file anggaran2024, apa isinya?", CORPUS)
        self.assertFalse(mention.found)
        self.assertIn("no document matches", mention.problem)
        self.assertIn("rehabGedung.pdf", mention.problem, "the available files are listed")


class OrdinaryQuestionsTests(unittest.TestCase):
    """`kontrak` and `dokumen` are ordinary words here. A question that merely
    contains one must be left exactly as it was."""

    def test_the_shipped_query_set_is_untouched(self) -> None:
        queries = [q["query"] for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]]
        scoped = [q for q in queries if (m := parse(q, CORPUS)).found or m.problem]
        self.assertEqual(scoped, [], "the gate's own queries name no document")

    def test_common_phrasings_name_no_document(self) -> None:
        for question in (
            "penyedia memutuskan kontrak secara sepihak",
            "daftar dokumen yang merupakan satu kesatuan bagian kontrak",
            "dalam kontrak ini apa sanksi keterlambatan?",
            "siapa para pihak dalam kontrak kerja konstruksi ini?",
            "berapa nilai kontrak?",
        ):
            with self.subTest(question=question):
                mention = parse(question, CORPUS)
                self.assertFalse(mention.found)
                self.assertEqual(mention.problem, "", "an ordinary question must not be stopped")


class CatalogueTests(unittest.TestCase):
    def test_the_catalogue_is_not_read_when_no_cue_appears(self) -> None:
        """It reads every row's metadata, and most questions name no document."""
        calls = []

        def provider():
            calls.append(1)
            return CORPUS

        parse("kewajiban penyedia mengasuransikan pekerjaan", provider)
        self.assertEqual(calls, [])

    def test_the_catalogue_is_read_at_most_once(self) -> None:
        calls = []

        def provider():
            calls.append(1)
            return CORPUS

        parse("pada file kontrak jasa, berapa dendanya?", provider)
        self.assertEqual(len(calls), 1)

    def test_nothing_is_inferred_without_filenames(self) -> None:
        """Views absent from disk: scoping still works by key through --document."""
        mention = parse("Pada file Rancangan Kontrak, siapa para pihak?", {"k1": "", "k2": ""})
        self.assertFalse(mention.found)
        self.assertEqual(mention.problem, "")


if __name__ == "__main__":
    unittest.main()
