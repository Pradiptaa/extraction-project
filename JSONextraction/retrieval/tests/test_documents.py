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

from retrieval.documents import parse
from retrieval.store import filename_resolver

CORPUS = {
    "k1": "Rancangan Kontrak.pdf",
    "k2": "kontrakJasa.pdf",
    "k3": "pembangunanRumah.pdf",
    "k4": "pembangunanSayap.pdf",
    "k5": "polres.pdf",
    "k6": "rehabGedung.pdf",
}

# Resolution is `store`'s job and is covered there; these tests are about which
# words in a question name a document, so they use the registry-free resolver.
RESOLVE = filename_resolver(CORPUS)

QUERY_SET = Path(__file__).resolve().parents[2] / "ground_truth" / "retrieval_queries.json"


class RecognisedTests(unittest.TestCase):
    def _scope(self, question: str) -> str:
        mention = parse(question, RESOLVE)
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
            parse("Pada file Rancangan Kontrak, siapa saja pihak yang terlibat?", RESOLVE).remainder,
            "siapa saja pihak yang terlibat?",
        )

    def test_a_name_running_into_the_question_takes_only_the_name(self) -> None:
        """No punctuation separates them, so the cut is by word, not to the end."""
        self.assertEqual(
            parse("di dokumen rehabGedung berapa masa pemeliharaan?", RESOLVE).remainder,
            "berapa masa pemeliharaan?",
        )

    def test_an_extension_is_stripped_with_the_name(self) -> None:
        self.assertEqual(
            parse("rehabGedung.pdf berapa lama masa pemeliharaan?", RESOLVE).remainder,
            "berapa lama masa pemeliharaan?",
        )


class RefusedTests(unittest.TestCase):
    def test_an_ambiguous_name_is_refused_with_the_candidates(self) -> None:
        mention = parse("Pada file kontrak, siapa para pihak?", RESOLVE)
        self.assertFalse(mention.found)
        self.assertIn("matches 2 documents", mention.problem)
        self.assertIn("Rancangan Kontrak.pdf", mention.problem)
        self.assertIn("kontrakJasa.pdf", mention.problem)

    def test_a_named_file_that_does_not_exist_is_reported(self) -> None:
        """`file` is never written by accident, so a miss is the user's answer,
        not a reason to quietly search every contract."""
        mention = parse("pada file anggaran2024, apa isinya?", RESOLVE)
        self.assertFalse(mention.found)
        self.assertIn("no document matches", mention.problem)

    def test_a_miss_does_not_print_the_catalogue(self) -> None:
        """Listing every document explained a miss while there were six. It
        explains nothing at a thousand, and the reply says where to look
        instead."""
        mention = parse("pada file anggaran2024, apa isinya?", RESOLVE)
        for filename in CORPUS.values():
            self.assertNotIn(filename, mention.problem)
        self.assertIn("--list-documents", mention.problem)

    def test_an_ambiguous_name_shows_a_few_candidates_and_a_count(self) -> None:
        crowd = {f"k{n}": f"kontrak{n}.pdf" for n in range(20)}
        mention = parse("pada file kontrak, siapa para pihak?", filename_resolver(crowd))
        self.assertFalse(mention.found)
        self.assertIn("matches 20 documents", mention.problem)
        self.assertLessEqual(len(mention.problem.splitlines()), 8, "a handful, not twenty")
        self.assertIn("more", mention.problem, "the rest are counted")


class OrdinaryQuestionsTests(unittest.TestCase):
    """`kontrak` and `dokumen` are ordinary words here. A question that merely
    contains one must be left exactly as it was."""

    def test_the_shipped_query_set_is_untouched(self) -> None:
        queries = [q["query"] for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]]
        scoped = [q for q in queries if (m := parse(q, RESOLVE)).found or m.problem]
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
                mention = parse(question, RESOLVE)
                self.assertFalse(mention.found)
                self.assertEqual(mention.problem, "", "an ordinary question must not be stopped")


class ResolverTests(unittest.TestCase):
    def _counting_resolver(self):
        calls = []
        inner = filename_resolver(CORPUS)

        def resolve(text, limit=5, filenames_only=False):
            calls.append(text)
            return inner(text, limit, filenames_only)

        return resolve, calls

    def test_nothing_is_resolved_when_no_cue_appears(self) -> None:
        """Most questions name no document, and resolution should cost nothing
        when there is nothing to resolve."""
        resolve, calls = self._counting_resolver()
        parse("kewajiban penyedia mengasuransikan pekerjaan", resolve)
        self.assertEqual(calls, [])

    def test_a_cue_resolves_only_the_words_that_could_be_a_name(self) -> None:
        resolve, calls = self._counting_resolver()
        parse("pada file kontrak jasa, berapa dendanya?", resolve)
        self.assertTrue(calls)
        self.assertTrue(all(len(text) < 40 for text in calls),
                        "the whole question is never offered as a name")


if __name__ == "__main__":
    unittest.main()


class RegistryBackedParseTests(unittest.TestCase):
    """`parse` against the registry, built from the real raw extractions.

    Every other test here uses the filename-only resolver, and that is how a
    serious defect got through: on the registry path a weak cue could match
    organisation names, and "dalam kontrak kerja konstruksi ini" was scoped to
    the contract whose organisation is "Satuan Kerja Dinas Tenaga Kerja". The
    tests passed because none of them ran the resolver `ask` actually uses.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import shutil
        import tempfile

        from retrieval import registry, store

        if not registry.RAW_DIR.is_dir() or not any(registry.RAW_DIR.glob("*_raw.json")):
            raise unittest.SkipTest("no raw extractions on disk")
        cls.tmp = Path(tempfile.mkdtemp())
        connection = registry.connect(registry.registry_path(cls.tmp))
        registry.rebuild(connection, "c")
        connection.close()
        cls.resolve = staticmethod(
            store._registry_resolver(cls.tmp, "c", lambda: filename_resolver({}))
        )
        cls._cleanup = staticmethod(lambda: shutil.rmtree(cls.tmp, ignore_errors=True))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._cleanup()

    def _scope(self, question: str):
        mention = parse(question, self.resolve)
        return next(iter(mention.scope.values())) if mention.found else None

    def test_ordinary_questions_are_left_alone(self) -> None:
        """Weak cues followed by words that appear in organisation names."""
        for question in (
            "siapa para pihak dalam kontrak kerja konstruksi ini?",
            "dalam kontrak pemerintah, apa kewajiban penyedia?",
            "pada dokumen dinas, siapa penandatangannya?",
            "kontrak kantor ini berlaku berapa lama?",
            "penyedia memutuskan kontrak secara sepihak",
            "daftar dokumen yang merupakan satu kesatuan bagian kontrak",
            "dalam kontrak ini apa sanksi keterlambatan?",
        ):
            with self.subTest(question=question):
                mention = parse(question, self.resolve)
                self.assertFalse(mention.found, f"scoped to {mention.scope}")
                self.assertEqual(mention.problem, "", "an ordinary question must not be stopped")

    def test_the_shipped_query_set_is_untouched(self) -> None:
        queries = [q["query"] for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]]
        touched = [q for q in queries if (m := parse(q, self.resolve)).found or m.problem]
        self.assertEqual(touched, [])

    def test_a_weak_cue_still_finds_a_filename(self) -> None:
        self.assertEqual(self._scope("dokumen polres, nomor kontraknya berapa?"), "polres.pdf")

    def test_a_strong_cue_may_match_a_contract_title(self) -> None:
        """Offered as a name, so the metadata is fair game: no filename holds
        "mekar", the title "Peningkatan Jalan Mekar ..." does."""
        self.assertEqual(self._scope("pada file mekar, siapa para pihak?"), "Rancangan Kontrak.pdf")

    def test_a_weak_cue_does_not_match_a_contract_title(self) -> None:
        self.assertIsNone(self._scope("dalam dokumen mekar, siapa para pihak?"))
