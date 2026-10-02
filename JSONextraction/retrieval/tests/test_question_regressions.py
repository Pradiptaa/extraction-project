from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from retrieval import registry
from retrieval.documents import parse
from retrieval.lookup import Route, lookup, raw_document_provider, route

RAW_DIR = Path(__file__).resolve().parents[2] / "output" / "raw"
SPECIMENS = ("Rancangan Kontrak", "kontrakJasa", "pembangunanRumah", "pembangunanSayap", "polres", "rehabGedung")
COLLECTION = "question_regressions"

SCOPED = [
    ("Pada dokumen dengan nomor 08/PUPRPRKP-B.PNK/SP-PPK, siapa pejabat yang menandatangani?",
     "Rancangan Kontrak.pdf", Route("parties")),
    ("Pada dokumen yang ditandatangani oleh Giajeng Wulandari, berapa nomor kontraknya?",
     "Rancangan Kontrak.pdf", Route("contract_number")),
    ("Berdasarkan dokumen Rancangan Kontrak, berapa nomor kontraknya?",
     "Rancangan Kontrak.pdf", Route("contract_number")),
    ("Berdasarkan dokumen yang ditandatangani oleh GIAJENG WULANDARI, berapa lama masa pemeliharaannya?",
     "Rancangan Kontrak.pdf", Route("key_numbers", "masa_pemeliharaan")),
    ("kontrak yang ditandatangani oleh ADI RUDINI, siapa penyedianya?",
     "pembangunanRumah.pdf", Route("parties")),
    ("dokumen polres, nomor kontraknya berapa?", "polres.pdf", Route("contract_number")),
    ("berapa lama masa pelaksanaannya di dokumen rehabGedung?",
     "rehabGedung.pdf", Route("key_numbers", "masa_pelaksanaan")),
    ("Masa pelaksanaannya berapa hari di dokumen polres?", "polres.pdf", Route("key_numbers", "masa_pelaksanaan")),
    ("kontrak yang ditandatangani oleh pejabat Dinas Pekerjaan Umum Bina Marga dan Cipta Karya, nomor kontraknya?",
     "rehabGedung.pdf", Route("contract_number")),
    ("dokumen yang ditandatangani oleh Mukhlisin, berapa lama masa pemeliharaan?",
     "pembangunanSayap.pdf", Route("key_numbers", "masa_pemeliharaan")),
]

ANSWERS = [
    ("Pada dokumen yang ditandatangani oleh Giajeng Wulandari, berapa nomor kontraknya?",
     ["08/PUPRPRKP-B.PNK/SP-PPK"]),
    ("Berdasarkan dokumen yang ditandatangani oleh GIAJENG WULANDARI, berapa lama masa pemeliharaannya?",
     ["180 hari kalender"]),
    ("Pada dokumen dengan nomor 08/PUPRPRKP-B.PNK/SP-PPK, siapa pejabat yang menandatangani?",
     ["GIAJENG WULANDARI", "Penyedia: (tidak terisi di dokumen)"]),
]


@unittest.skipUnless(all((RAW_DIR / f"{s}_raw.json").exists() for s in SPECIMENS),
                     "needs output/raw — run the gate, which regenerates it")
class QuestionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.connection = registry.connect(cls.tmp / "registry.sqlite3")
        registry.rebuild(cls.connection, COLLECTION, raw_dir=RAW_DIR)
        cls.raw = staticmethod(raw_document_provider(raw_dir=RAW_DIR))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _resolve(self, text: str, limit: int = 5, filenames_only: bool = False):
        return registry.resolve(self.connection, text, COLLECTION, limit, filenames_only)

    def _ask(self, question: str):
        mention = parse(question, self._resolve)
        return mention, route(mention.remainder if mention.found else question)

    def test_each_question_is_scoped_to_its_document_and_routed_to_its_field(self) -> None:
        for question, filename, expected_route in SCOPED:
            with self.subTest(question=question):
                mention, got_route = self._ask(question)
                self.assertEqual(mention.problem, "")
                self.assertEqual(list(mention.scope.values()), [filename])
                self.assertEqual(got_route, expected_route)

    def test_the_answer_comes_from_that_document_only_and_holds_only_its_own_value(self) -> None:
        for question, expected in ANSWERS:
            with self.subTest(question=question):
                mention, got_route = self._ask(question)
                answers = lookup(got_route, mention.scope, self.raw)
                self.assertEqual([a.name for a in answers], list(mention.scope.values()))
                (answer,) = answers
                self.assertEqual(answer.status, "found")
                self.assertEqual(len(answer.lines), len(expected), answer.lines)
                for line, value in zip(answer.lines, expected):
                    self.assertIn(value, line)

    def test_a_name_shared_by_several_documents_is_refused_with_its_candidates(self) -> None:
        mention, _ = self._ask("Siapa PPK di berkas Dinas Pekerjaan Umum?")
        self.assertFalse(mention.found)
        self.assertIn("matches 3 documents", mention.problem)

    def test_an_unknown_signatory_is_refused_rather_than_searched_everywhere(self) -> None:
        mention, _ = self._ask("Pada dokumen yang ditandatangani oleh Siti Aminah, berapa nilai kontraknya?")
        self.assertFalse(mention.found)
        self.assertIn("no document matches", mention.problem)

    def test_a_description_that_names_no_one_continues_unscoped(self) -> None:
        mention, got_route = self._ask("dokumen yang ditandatangani oleh kepala dinas terkait, berapa nomor kontraknya?")
        self.assertFalse(mention.found)
        self.assertEqual(mention.problem, "")
        self.assertEqual(got_route, Route("contract_number"))

    def test_a_field_question_without_a_document_is_routed_but_not_scoped(self) -> None:
        for question, expected_route in (("berapa lama masa pemeliharaannya?", Route("key_numbers", "masa_pemeliharaan")),
                                         ("siapa penyedianya?", Route("parties")),
                                         ("berapa nomor kontraknya?", Route("contract_number"))):
            with self.subTest(question=question):
                mention, got_route = self._ask(question)
                self.assertFalse(mention.found)
                self.assertEqual(mention.problem, "")
                self.assertEqual(got_route, expected_route)


if __name__ == "__main__":
    unittest.main()
