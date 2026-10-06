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

    def test_a_field_name_containing_a_cue_does_not_hide_the_real_cue(self) -> None:
        self.assertEqual(self._scope("Berapa nomor kontrak dokumen Rancangan?"),
                         "Rancangan Kontrak.pdf")

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
        self.assertEqual(self._scope("pada file pembangunan sayap, apa ruang lingkupnya?"),
                         "pembangunanSayap.pdf")


class RemainderTests(unittest.TestCase):

    def test_the_question_survives_the_strip(self) -> None:
        self.assertEqual(
            parse("Pada file Rancangan Kontrak, siapa saja pihak yang terlibat?", RESOLVE).remainder,
            "siapa saja pihak yang terlibat?",
        )

    def test_a_name_running_into_the_question_takes_only_the_name(self) -> None:
        self.assertEqual(
            parse("di dokumen rehabGedung berapa masa pemeliharaan?", RESOLVE).remainder,
            "berapa masa pemeliharaan?",
        )

    def test_an_extension_is_stripped_with_the_name(self) -> None:
        self.assertEqual(
            parse("rehabGedung.pdf berapa lama masa pemeliharaan?", RESOLVE).remainder,
            "berapa lama masa pemeliharaan?",
        )

    def test_the_question_before_a_filename_survives(self) -> None:
        self.assertEqual(
            parse("Berapa nilai kontrak di Rancangan Kontrak.pdf?", RESOLVE).remainder,
            "Berapa nilai kontrak?",
        )

    def test_a_name_in_brackets_leaves_no_empty_brackets(self) -> None:
        self.assertEqual(
            parse("Berapa masa pemeliharaan (kontrak polres)?", RESOLVE).remainder,
            "Berapa masa pemeliharaan?",
        )

    def test_a_question_that_is_only_a_name_keeps_its_words(self) -> None:
        mention = parse("file polres", RESOLVE)
        self.assertEqual(mention.scope, {"k5": "polres.pdf"})
        self.assertEqual(mention.remainder, "file polres", "an empty query retrieves at random")


class SeveralCuesTests(unittest.TestCase):
    def test_a_named_file_after_one_that_matches_nothing_is_found(self) -> None:
        mention = parse("Di file mana tercantum denda, berkas polres?", RESOLVE)
        self.assertEqual(mention.scope, {"k5": "polres.pdf"})
        self.assertEqual(mention.remainder, "Di file mana tercantum denda?")

    def test_two_named_documents_are_both_scoped(self) -> None:
        mention = parse("Bandingkan denda kontrak polres dan kontrak rehab gedung?", RESOLVE)
        self.assertEqual(set(mention.scope.values()), {"polres.pdf", "rehabGedung.pdf"})
        self.assertEqual(mention.remainder, "Bandingkan denda?")

    def test_a_cue_followed_by_a_colon_is_a_cue(self) -> None:
        self.assertEqual(parse("dokumen: polres, berapa nilainya?", RESOLVE).scope,
                         {"k5": "polres.pdf"})

    def test_a_field_name_before_the_cue_stays_in_the_question(self) -> None:
        for question, remainder in (
            ("Nomor kontrak polres berapa?", "Nomor kontrak berapa?"),
            ("Berapa nilai kontrak polres?", "Berapa nilai kontrak?"),
            ("Tanggal kontrak dokumen rehab?", "Tanggal kontrak?"),
            ("Berapa masa pemeliharaan kontrak polres?", "Berapa masa pemeliharaan?"),
        ):
            with self.subTest(question=question):
                self.assertEqual(parse(question, RESOLVE).remainder, remainder)


class RefusedTests(unittest.TestCase):
    def test_an_ambiguous_name_is_refused_with_the_candidates(self) -> None:
        mention = parse("Pada file kontrak, siapa para pihak?", RESOLVE)
        self.assertFalse(mention.found)
        self.assertIn("cocok dengan 2 dokumen", mention.problem)
        self.assertIn("Rancangan Kontrak.pdf", mention.problem)
        self.assertIn("kontrakJasa.pdf", mention.problem)

    def test_a_named_file_that_does_not_exist_is_reported(self) -> None:
        mention = parse("pada file anggaran2024, apa isinya?", RESOLVE)
        self.assertFalse(mention.found)
        self.assertIn("Tidak ada dokumen bernama", mention.problem)

    def test_a_miss_does_not_print_the_catalogue(self) -> None:
        mention = parse("pada file anggaran2024, apa isinya?", RESOLVE)
        for filename in CORPUS.values():
            self.assertNotIn(filename, mention.problem)
        self.assertNotIn("--", mention.problem)

    def test_an_ambiguous_name_shows_a_few_candidates_and_a_count(self) -> None:
        crowd = {f"k{n}": f"kontrak{n}.pdf" for n in range(20)}
        mention = parse("pada file kontrak, siapa para pihak?", filename_resolver(crowd))
        self.assertFalse(mention.found)
        self.assertIn("cocok dengan 20 dokumen", mention.problem)
        self.assertLessEqual(len(mention.problem.splitlines()), 8, "a handful, not twenty")
        self.assertIn("15 lainnya", mention.problem, "the rest are counted")


class SignatoryTests(unittest.TestCase):
    SIGNERS = {"giajeng wulandari": ("k1", "Rancangan Kontrak.pdf"), "adi rudini": ("k3", "pembangunanRumah.pdf"),
               "budi": ("k2", "kontrakJasa.pdf"), "budi santoso": ("k4", "polres.pdf")}

    def _resolve(self, text, limit=5, filenames_only=False):
        from retrieval.registry import Document, Matches

        if filenames_only:
            return Matches()
        wanted = " ".join(text.lower().split())
        hits = [Document(document_key=key, filename=name) for person, (key, name) in self.SIGNERS.items()
                if person.startswith(wanted)]
        return Matches(documents=hits[:limit], total=len(hits))

    def test_a_signatory_names_their_document(self) -> None:
        mention = parse("Berdasarkan dokumen yang ditandatangani oleh GIAJENG WULANDARI, berapa lama masa "
                        "pemeliharaannya?", self._resolve)
        self.assertEqual(mention.scope, {"k1": "Rancangan Kontrak.pdf"})
        self.assertEqual(mention.remainder, "berapa lama masa pemeliharaannya?")

    def test_other_signing_verbs_and_cues_work(self) -> None:
        for question in ("kontrak yang disetujui oleh Adi Rudini, siapa penyedianya?",
                         "siapa penyedia dalam perjanjian yang dibuat oleh adi rudini?"):
            with self.subTest(question=question):
                self.assertEqual(parse(question, self._resolve).scope, {"k3": "pembangunanRumah.pdf"})

    def test_an_unknown_signatory_is_reported_not_searched_everywhere(self) -> None:
        mention = parse("dokumen yang ditandatangani oleh Siti Aminah, berapa nilainya?", self._resolve)
        self.assertFalse(mention.found)
        self.assertIn("Tidak ada dokumen bernama", mention.problem)

    def test_a_description_after_the_cue_is_not_refused(self) -> None:
        for question in ("dokumen yang ditandatangani oleh pejabat dinas terkait, berapa nilainya?",
                         "kontrak yang dibuat oleh kepala bidang, siapa penyedianya?"):
            with self.subTest(question=question):
                mention = parse(question, self._resolve)
                self.assertFalse(mention.found)
                self.assertEqual(mention.problem, "")

    def test_a_name_inside_a_description_is_still_found(self) -> None:
        mention = parse("dokumen yang ditandatangani oleh pejabat PPK Adi Rudini, siapa penyedianya?", self._resolve)
        self.assertEqual(mention.scope, {"k3": "pembangunanRumah.pdf"})

    def test_filler_words_alone_never_scope(self) -> None:
        self.SIGNERS = dict(self.SIGNERS, **{"dinas": ("k9", "x.pdf")})
        mention = parse("dokumen yang ditandatangani oleh pejabat dinas, berapa nilainya?", self._resolve)
        self.assertFalse(mention.found)

    def test_a_signatory_on_several_documents_is_refused_with_the_candidates(self) -> None:
        mention = parse("dokumen yang ditandatangani oleh Budi, berapa nilainya?", self._resolve)
        self.assertFalse(mention.found)
        self.assertIn("cocok dengan 2 dokumen", mention.problem)


class OrdinaryQuestionsTests(unittest.TestCase):

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
            "Apa urutan dokumen kontrak?",
            "Apa saja dokumen kontrak yang berlaku?",
            "Berapa jaminan pelaksanaan kontrak pembangunan?",
            "kontrak re?",
            "Nomor kontrak: berapa?",
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
        resolve, calls = self._counting_resolver()
        parse("kewajiban penyedia mengasuransikan pekerjaan", resolve)
        self.assertEqual(calls, [])

    def test_a_cue_resolves_only_the_words_that_could_be_a_name(self) -> None:
        resolve, calls = self._counting_resolver()
        parse("pada file kontrak jasa, berapa dendanya?", resolve)
        self.assertTrue(calls)
        self.assertTrue(all(len(text) < 40 for text in calls),
                        "the whole question is never offered as a name")


class ExactNameTests(unittest.TestCase):
    RESOLVE = staticmethod(filename_resolver({**CORPUS, "k7": "Rancangan Kontrak ABC.pdf"}))

    def test_the_exact_name_picks_the_shorter_contract(self) -> None:
        mention = parse("Pada file Rancangan Kontrak, apa isinya?", self.RESOLVE)
        self.assertEqual(list(mention.scope.values()), ["Rancangan Kontrak.pdf"])

    def test_the_longer_name_still_picks_the_longer_contract(self) -> None:
        mention = parse("Pada file Rancangan Kontrak ABC, apa isinya?", self.RESOLVE)
        self.assertEqual(list(mention.scope.values()), ["Rancangan Kontrak ABC.pdf"])

    def test_a_fragment_shared_by_both_is_still_ambiguous(self) -> None:
        mention = parse("Pada file Rancangan, apa isinya?", self.RESOLVE)
        self.assertFalse(mention.found)
        self.assertIn("cocok dengan 2 dokumen", mention.problem)


class DroppedWordTests(unittest.TestCase):
    def test_an_unmatched_word_after_the_name_is_reported(self) -> None:
        mention = parse("Pada file Rancangan Kontrak 2099, apa isinya?", RESOLVE)
        self.assertEqual(list(mention.scope.values()), ["Rancangan Kontrak.pdf"])
        self.assertEqual(mention.note,
                         "Tidak ada dokumen bernama \"Rancangan Kontrak 2099\" — menggunakan Rancangan Kontrak.pdf.")

    def test_question_words_after_the_name_are_not_reported(self) -> None:
        for question in ("pada file rehabGedung apa isinya?", "Pada file rehabGedung Apa isinya?",
                         "Pada file Rancangan Kontrak, apa isinya?"):
            with self.subTest(question=question):
                mention = parse(question, RESOLVE)
                self.assertTrue(mention.found)
                self.assertEqual(mention.note, "")


if __name__ == "__main__":
    unittest.main()


class RegistryBackedParseTests(unittest.TestCase):

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

    def test_the_shipped_query_set_is_untouched_but_its_contract_number(self) -> None:
        queries = {q["id"]: q["query"] for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]}
        number = queries.pop("q19_identifier_nomor_kontrak")
        touched = [q for q in queries.values() if (m := parse(q, self.resolve)).found or m.problem]
        self.assertEqual(touched, [])
        mention = parse(number, self.resolve)
        self.assertEqual(list(mention.scope.values()), ["Rancangan Kontrak.pdf"])
        self.assertEqual(mention.remainder, number)

    def test_a_weak_cue_still_finds_a_filename(self) -> None:
        self.assertEqual(self._scope("dokumen polres, nomor kontraknya berapa?"), "polres.pdf")

    def test_a_strong_cue_may_match_a_contract_title(self) -> None:
        self.assertEqual(self._scope("pada file mekar, siapa para pihak?"), "Rancangan Kontrak.pdf")

    def test_a_weak_cue_does_not_match_a_contract_title(self) -> None:
        self.assertIsNone(self._scope("dalam dokumen mekar, siapa para pihak?"))

    def test_a_contract_number_is_not_cut_at_its_full_stop(self) -> None:
        mention = parse("Kapan tanggal di file 08/PUPRPRKP-B.PNK/SP-PPK?", self.resolve)
        self.assertEqual(next(iter(mention.scope.values())), "Rancangan Kontrak.pdf")
        self.assertEqual(mention.remainder, "Kapan tanggal?")

    def test_a_bare_filename_does_not_swallow_the_question(self) -> None:
        mention = parse("Berapa nilai kontrak di Rancangan Kontrak.pdf?", self.resolve)
        self.assertEqual(mention.remainder, "Berapa nilai kontrak?")

    def test_a_contract_number_names_its_document_without_file(self) -> None:
        for question, remainder in (
            ("Pada dokumen dengan nomor kontrak 08/PUPRPRKP-B.PNK/SP-PPK, Siapa pejabat yang menandatangani?",
             "Siapa pejabat yang menandatangani?"),
            ("No. Kontrak: 08/PUPRPRKP-B.PNK/SP-PPK, berapa nilainya?", "berapa nilainya?"),
            ("Berapa denda pada kontrak nomor 08/PUPRPRKP-B.PNK/SP-PPK?", "Berapa denda?"),
            ("Pada dokumen dengan nomor 08/PUPRPRKP-B.PNK/SP-PPK, Siapa pejabat yang menandatangani?",
             "Siapa pejabat yang menandatangani?"),
            ("Siapa pejabat yang menandatangani 08/PUPRPRKP-B.PNK/SP-PPK?",
             "Siapa pejabat yang menandatangani?"),
            ("Berapa nilai kontrak dengan no. 08/PUPRPRKP-B.PNK/SP-PPK?", "Berapa nilai kontrak?"),
        ):
            with self.subTest(question=question):
                mention = parse(question, self.resolve)
                self.assertEqual(next(iter(mention.scope.values()), None), "Rancangan Kontrak.pdf")
                self.assertEqual(mention.remainder, remainder)

    def test_an_unknown_contract_number_searches_every_document(self) -> None:
        mention = parse("Siapa penyedia untuk nomor kontrak 99/XYZ/2020?", self.resolve)
        self.assertFalse(mention.found)
        self.assertEqual(mention.problem, "")

    def test_a_contract_number_does_not_stop_a_question_without_the_registry(self) -> None:
        mention = parse("Siapa penyedia untuk nomor kontrak 08/PUPRPRKP-B.PNK/SP-PPK?", RESOLVE)
        self.assertEqual(mention.problem, "")

    def test_numbers_that_are_not_contract_numbers_scope_nothing(self) -> None:
        for question in (
            "Apa isi Perpres 16/2018?",
            "Kontrak ditandatangani 12/03/2021, berapa nilainya?",
            "Nilai kontrak Rp 1.500.000, berapa denda?",
            "Apa isi Pasal 5 ayat (3)?",
        ):
            with self.subTest(question=question):
                mention = parse(question, self.resolve)
                self.assertFalse(mention.found, f"scoped to {mention.scope}")
                self.assertEqual(mention.problem, "")

    def test_a_number_without_separators_is_not_a_contract_number(self) -> None:
        mention = parse("Apa nomor kontrak 2 tahun lalu?", self.resolve)
        self.assertFalse(mention.found)
        self.assertEqual(mention.problem, "")
