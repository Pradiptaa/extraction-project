"""`retrieval.lookup` — routing core-field questions and answering them from a raw extraction.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from retrieval.lookup import Route, has_answer, load_raw_documents, lookup, render, route

KEY_A, KEY_B = "aaaa1111", "bbbb2222"


def _core(**fields) -> dict:
    base = {
        "contract_name": {"value": None, "confidence": 0.0, "flags": []},
        "contract_number": {"value": None, "confidence": 0.0, "flags": ["review_required"]},
        "parties": {"value": []},
        "key_dates": {"value": []},
        "key_numbers": {"value": []},
    }
    base.update(fields)
    return base


FILLED = {
    "source": {"sha256": KEY_A, "file": "a.pdf"},
    "core": _core(
        contract_name={"value": "Pembangunan Kantor Polres", "confidence": 0.94, "flags": ["ambiguous"]},
        contract_number={"value": "08/PUPR/SP-PPK", "confidence": 0.88, "flags": []},
        parties={"value": [
            {"role_label": "Pejabat Penandatangan\nKontrak", "organization": {"value": "Dinas PU"},
             "representative": {"name": "BUDI, ST", "position": "PPK"}, "flags": []},
            {"role_label": "Penyedia", "organization": {"value": None},
             "representative": {"name": None}, "flags": ["template_placeholder"]},
        ]},
        key_dates={"value": [
            {"date": "2017", "raw": "Nomor 2 Tahun 2017", "precision": "year"},
            {"date": "2023-04-03", "raw": "03 April 2023", "precision": "day"},
            {"date": "2023-04-03", "raw": "3 April 2023", "precision": "day"},
        ]},
        key_numbers={"value": [
            {"type": "duration", "subtype": "masa_pemeliharaan", "amount": 180, "unit": "hari_kalender",
             "raw": "180 (seratus delapan puluh) hari kalender", "confidence": 0.95},
            {"type": "duration", "subtype": "masa_pemeliharaan", "amount": 180, "unit": "hari_kalender",
             "raw": "180 hari kalender", "confidence": 0.95},
            {"type": "monetary", "subtype": "unclassified", "amount": 25e9, "currency": "IDR",
             "raw": "Rp25.000.000.000,00", "confidence": 0.5},
            {"type": "penalty_rate", "subtype": "denda_keterlambatan", "amount": 0.001, "unit": "ratio",
             "raw": "1/1000", "confidence": 0.85},
            {"type": "contract_value", "amount": None, "currency": "IDR", "raw": "Rp. ....,00",
             "confidence": 0.0, "flags": ["template_placeholder"]},
        ]},
    ),
}
TEMPLATE = {
    "source": {"sha256": KEY_B, "file": "b.pdf"},
    "core": _core(contract_name={"value": "........ [diisi nama paket pekerjaan]", "confidence": 0.94, "flags": []}),
}
RAW = {KEY_A: (Path("a_raw.json"), FILLED), KEY_B: (Path("b_raw.json"), TEMPLATE)}
SCOPE = {KEY_A: "a.pdf", KEY_B: "b.pdf"}


class RouterTests(unittest.TestCase):
    def test_core_field_questions_route_to_their_field(self) -> None:
        cases = {
            "nama kontrak": Route("contract_name"),
            "Nomor kontrak?": Route("contract_number"),
            "siapa penyedia?": Route("parties"),
            "nama perusahaan penyedia": Route("parties"),
            "tanggal penting kontrak": Route("key_dates"),
            "berapa nilai kontrak?": Route("key_numbers", "contract_value"),
            "berapa lama masa pemeliharaan?": Route("key_numbers", "masa_pemeliharaan"),
            "berapa denda keterlambatan?": Route("key_numbers", "denda_keterlambatan"),
            "angka penting": Route("key_numbers"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(route(question), expected)

    def test_clause_questions_go_to_search(self) -> None:
        """A wrong quick answer is worse than a slower right one."""
        for question in (
            "kewajiban penyedia mengasuransikan pekerjaan",
            "asuransi pihak ketiga",
            "bagaimana jika nilai kontrak berubah?",
            "hak para pihak",
            "alamat korespondensi para pihak",
            "keadaan kahar",
        ):
            with self.subTest(question=question):
                self.assertIsNone(route(question))

    def test_a_bare_topic_without_a_quantity_word_is_searched(self) -> None:
        self.assertIsNone(route("masa pemeliharaan"))
        self.assertIsNone(route("denda keterlambatan"))

    def test_long_questions_are_searched(self) -> None:
        self.assertIsNone(route("tolong sebutkan nomor kontrak yang tercantum pada halaman pertama dokumen perjanjian ini ya"))


class LookupTests(unittest.TestCase):
    def _lines(self, target: Route, key: str = KEY_A):
        (answer,) = lookup(target, {key: SCOPE[key]}, RAW)
        return answer

    def test_scalar_field_carries_its_caveat(self) -> None:
        answer = self._lines(Route("contract_name"))
        self.assertEqual(answer.lines, ["Pembangunan Kantor Polres  (Need Review)"])

    def test_confident_value_has_no_review_marker(self) -> None:
        self.assertEqual(self._lines(Route("contract_number")).lines, ["08/PUPR/SP-PPK"])

    def test_template_blank_is_unfilled_not_a_miss(self) -> None:
        self.assertEqual(self._lines(Route("contract_name"), KEY_B).status, "unfilled")

    def test_null_field_is_unresolved(self) -> None:
        self.assertEqual(self._lines(Route("contract_number"), KEY_B).status, "unresolved")

    def test_parties_show_filled_and_blank_roles_on_one_line_each(self) -> None:
        answer = self._lines(Route("parties"))
        self.assertEqual(answer.lines, [
            "Pejabat Penandatangan Kontrak: Dinas PU — diwakili BUDI, ST, PPK",
            "Penyedia: (tidak terisi di dokumen)",
        ])

    def test_dates_skip_year_only_citations_and_duplicates(self) -> None:
        answer = self._lines(Route("key_dates"))
        self.assertEqual([line for line in answer.lines if line.startswith("20")],
                         ['2023-04-03  (teks: "03 April 2023")'])

    def test_number_subtype_is_deduplicated_and_formatted(self) -> None:
        answer = self._lines(Route("key_numbers", "masa_pemeliharaan"))
        self.assertEqual(len(answer.lines), 1)
        self.assertIn("180 hari kalender", answer.lines[0])

    def test_placeholder_contract_value_is_unfilled(self) -> None:
        self.assertEqual(self._lines(Route("key_numbers", "contract_value")).status, "unfilled")

    def test_important_numbers_exclude_unclassified_amounts(self) -> None:
        text = "\n".join(self._lines(Route("key_numbers")).lines)
        self.assertIn("Denda keterlambatan: 0.1%", text)
        self.assertNotIn("Rp25.000.000.000", text)

    def test_a_document_without_a_raw_file_is_reported(self) -> None:
        (answer,) = lookup(Route("contract_name"), {"cccc3333": "c.pdf"}, RAW)
        self.assertEqual(answer.status, "missing_raw")

    def test_nothing_usable_anywhere_means_no_answer(self) -> None:
        answers = lookup(Route("contract_number"), {KEY_B: "b.pdf"}, RAW)
        self.assertFalse(has_answer(answers))
        self.assertTrue(has_answer(lookup(Route("contract_name"), {KEY_B: "b.pdf"}, RAW)))

    def test_render_names_every_document_and_the_source_field(self) -> None:
        text = render(Route("contract_name"), lookup(Route("contract_name"), SCOPE, RAW))
        self.assertIn("a.pdf:", text)
        self.assertIn("b.pdf:", text)
        self.assertIn("Sumber: core.contract_name — a_raw.json, b_raw.json", text)

    def test_raw_files_load_by_source_sha_and_ignore_other_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a_raw.json").write_text(json.dumps(FILLED), encoding="utf-8")
            Path(tmp, "legacy.json").write_text(json.dumps(TEMPLATE), encoding="utf-8")
            self.assertEqual(list(load_raw_documents(Path(tmp))), [KEY_A])


if __name__ == "__main__":
    unittest.main()
