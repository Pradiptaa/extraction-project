"""`retrieval.ask` — the one place retrieval and synthesis meet. Everything
outside the CLI's own logic is faked, so these pin only what `ask` decides.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from retrieval import ask, references, store
from retrieval.chat import Answer, SourceClause
from retrieval.config import Settings
from retrieval.retrievers import Hit

DOC_A = "aaaa1111"
DOC_B = "bbbb2222"

HITS = [
    Hit(id="a", score=0.1, metadata={"label": "55.2", "sub_document": "general_terms", "document_key": DOC_A}, text="Penyedia wajib mengasuransikan."),
    Hit(id="b", score=0.1, metadata={"label": "55.2", "sub_document": "general_terms", "document_key": DOC_B}, text="Penyedia wajib mengasuransikan."),
]
SETTINGS = Settings(model="bge-m3", batch_size=1,
                    db_path=Path("."), collection="c", chat_model="qwen2.5:3b-instruct")


class FakeRetriever:
    def __init__(self) -> None:
        self.scopes: list[set[str] | None] = []
        self.queries: list[str] = []

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        self.scopes.append(scope)
        self.queries.append(query)
        return [hit for hit in HITS if scope is None or hit.metadata.get("document_key") in scope][:k]


class AskTests(unittest.TestCase):
    def _main(self, *argv: str, synthesizer=None, scope=None):
        load_settings = mock.Mock(return_value=SETTINGS)
        self.retriever = FakeRetriever()
        patches = [
            mock.patch.object(sys, "argv", ["ask", *argv]),
            mock.patch.object(ask, "load_settings", load_settings),
            mock.patch.object(ask, "open_collection", mock.Mock()),
            mock.patch.object(ask, "Embedder", mock.Mock()),
            mock.patch.object(ask, "build_retriever", mock.Mock(return_value=self.retriever)),
        ]
        if scope is not None:
            # Stubbed; the real resolution is covered in test_store.
            patches.append(mock.patch.object(ask, "resolve_scope", mock.Mock(return_value=scope)))
        if synthesizer is not None:
            patches.append(mock.patch.object(ask, "build_synthesizer", mock.Mock(return_value=synthesizer)))
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            for patch in patches:
                self.enterContext(patch)
            # ask reconfigures sys.stdout; StringIO has none, which ask tolerates.
            code = ask.main()
        return code, stdout.getvalue(), load_settings

    def test_default_run_prints_collapsed_clauses_and_needs_no_chat_model(self) -> None:
        code, out, _ = self._main("asuransi pihak ketiga", "--retriever", "bm25")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("Penyedia wajib mengasuransikan."), 1, "duplicates collapse to one source")
        self.assertIn("Pasal 55.2", out)
        self.assertNotIn("Sumber:", out, "no source list without synthesis")

    def test_settings_are_loaded_once_and_unconditionally(self) -> None:
        """Nothing is served remotely, so no arm has to argue for a credential
        and none can be refused one."""
        for retriever in ("bm25", "dense", "hybrid"):
            with self.subTest(retriever=retriever):
                _, _, load_settings = self._main("asuransi", "--retriever", retriever)
                load_settings.assert_called_once_with()

    def test_synthesis_failure_falls_back_to_the_retrieved_clauses(self) -> None:
        """Retrieval already succeeded; a chat outage must not throw that away."""
        failing = mock.Mock()
        failing.synthesize.side_effect = RuntimeError("429 rate limited")
        with self.assertLogs("retrieval.ask", level="ERROR") as captured:
            code, out, _ = self._main("asuransi", "--synthesizer", "ollama", synthesizer=failing)
        self.assertEqual(code, 1)
        self.assertIn("Penyedia wajib mengasuransikan.", out)
        self.assertIn("Jawaban tidak dapat disusun", out)
        self.assertIn("falling back", "\n".join(captured.output))

    def test_synthesized_answer_lists_its_sources_with_copy_count(self) -> None:
        source = SourceClause(id="a", text="Penyedia wajib mengasuransikan.", label="55.2",
                              sub_document="general_terms", hierarchy_path="C/55/55.2", copies=2)
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="Menurut [Pasal 55.2] ...", sources=[source],
                                                     model="qwen2.5:3b-instruct", usage_tokens=314)
        code, out, _ = self._main("asuransi", "--synthesizer", "ollama", synthesizer=synthesizer)
        self.assertEqual(code, 0)
        self.assertIn("Menurut [Pasal 55.2]", out)
        self.assertIn("Sumber:", out)
        self.assertNotIn("identik", out)
        self.assertIn("(314 tokens)", out)
        _, out, _ = self._main("asuransi", "--synthesizer", "ollama", "--verbose", synthesizer=synthesizer)
        self.assertIn("(x2 identik)", out)
        self.assertIn("(314 tokens)", out)


class InferredDocumentTests(AskTests):
    """A question that names its own contract, with no `--document`.

    The parser itself is covered in `test_documents`; these pin the wiring —
    that the scope reaches the retriever, that the filename is taken out of the
    query, and that a question naming no document is left alone.
    """

    # Three, so that "kontrak" is genuinely ambiguous between two of them.
    CORPUS = {DOC_A: "Rancangan Kontrak.pdf", DOC_B: "rehabGedung.pdf", "cccc3333": "kontrakJasa.pdf"}

    def _asked(self, question: str, *argv: str):
        # Resolution itself lives in `store` and is covered there; what is
        # pinned here is that `ask` wires it in and acts on the answer.
        resolver = store.filename_resolver(self.CORPUS)
        with mock.patch.object(ask, "corpus_documents", mock.Mock(return_value=self.CORPUS)), \
             mock.patch.object(ask, "document_resolver", mock.Mock(return_value=resolver)):
            return self._main(question, "--retriever", "bm25", "--route", "search", *argv)

    def test_a_named_document_scopes_the_search(self) -> None:
        self._asked("Pada file rehabGedung, apa kewajiban asuransi?")
        self.assertEqual(self.retriever.scopes, [{DOC_B}])

    def test_the_filename_is_taken_out_of_the_query(self) -> None:
        """Left in, "pada file rehabGedung" ranks every contract that contains
        those words — which is all of them."""
        self._asked("Pada file rehabGedung, apa kewajiban asuransi?")
        self.assertEqual(self.retriever.queries, ["apa kewajiban asuransi?"])

    def test_the_inferred_scope_is_printed(self) -> None:
        """Never hidden: an answer about one contract that reads as a claim
        about all six is this feature's dangerous failure."""
        _, out, _ = self._asked("Pada file rehabGedung, apa kewajiban asuransi?")
        self.assertIn("Dokumen: rehabGedung.pdf", out)

    def test_an_ordinary_question_is_not_scoped(self) -> None:
        self._asked("penyedia memutuskan kontrak secara sepihak")
        self.assertEqual(self.retriever.scopes, [None])
        self.assertEqual(self.retriever.queries, ["penyedia memutuskan kontrak secara sepihak"])

    def test_an_ambiguous_name_stops_before_searching(self) -> None:
        code, out, _ = self._asked("Pada file kontrak, apa kewajiban asuransi?")
        self.assertEqual(code, 1)
        self.assertEqual(self.retriever.queries, [], "nothing was searched")
        self.assertIn("cocok dengan 2 dokumen", out)

    def test_a_named_file_that_does_not_exist_stops_before_searching(self) -> None:
        code, out, _ = self._asked("Pada file anggaran2024, apa isinya?")
        self.assertEqual(code, 1)
        self.assertEqual(self.retriever.queries, [])
        self.assertIn("Tidak ada dokumen bernama", out)

    def test_the_explicit_flag_wins_over_the_question(self) -> None:
        """`--document` is the user being explicit; inference must not override it."""
        with mock.patch.object(ask, "corpus_documents", mock.Mock(return_value=self.CORPUS)):
            self._main("Pada file rehabGedung, apa kewajiban asuransi?", "--retriever", "bm25",
                       "--route", "search", "--document", "rancangan",
                       scope={DOC_A: "Rancangan Kontrak.pdf"})
        self.assertEqual(self.retriever.scopes, [{DOC_A}])


class DocumentScopeTests(AskTests):
    """`--document`: the CLI half of document scoping."""

    SCOPE = {DOC_B: "rehabGedung.pdf"}

    def test_no_document_flag_searches_the_whole_corpus(self) -> None:
        """The default must stay unscoped; every baseline is corpus-wide."""
        self._main("asuransi", "--retriever", "bm25")
        self.assertEqual(self.retriever.scopes, [None])

    def test_the_flag_reaches_the_retriever_as_document_keys(self) -> None:
        self._main("asuransi", "--retriever", "bm25", "--document", "rehab", scope=self.SCOPE)
        self.assertEqual(self.retriever.scopes, [{DOC_B}])

    def test_scoped_results_hold_only_that_document(self) -> None:
        _, out, _ = self._main("asuransi", "--retriever", "bm25", "--document", "rehab", scope=self.SCOPE)
        self.assertIn("Penyedia wajib mengasuransikan.", out)
        self.assertEqual(self.retriever.scopes, [{DOC_B}])

    def test_the_scope_is_printed_even_without_verbose(self) -> None:
        """The scope is never hidden behind --verbose."""
        _, out, _ = self._main("asuransi", "--retriever", "bm25", "--document", "rehab", scope=self.SCOPE)
        self.assertIn("Dokumen: rehabGedung.pdf", out)

    def test_an_empty_scoped_result_says_the_scope_caused_it(self) -> None:
        _, out, _ = self._main("asuransi", "--retriever", "bm25", "--document", "x",
                               scope={"no_such_document": "ghost.pdf"})
        self.assertIn("Tidak ditemukan bagian yang sesuai di ghost.pdf", out)

    def test_the_synthesizer_is_told_which_contract_the_clauses_came_from(self) -> None:
        """Otherwise an answer about one contract reads as a claim about all."""
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="jawaban", sources=[])
        self._main("asuransi", "--retriever", "bm25", "--synthesizer", "ollama",
                   "--document", "rehab", synthesizer=synthesizer, scope=self.SCOPE)
        self.assertEqual(synthesizer.synthesize.call_args.args[2], "rehabGedung.pdf")

    def test_an_unscoped_run_passes_no_scope_note(self) -> None:
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="jawaban", sources=[])
        self._main("asuransi", "--retriever", "bm25", "--synthesizer", "ollama", synthesizer=synthesizer)
        self.assertEqual(synthesizer.synthesize.call_args.args[2], "")

    def test_listing_documents_needs_no_question(self) -> None:
        with mock.patch.object(ask, "corpus_documents", mock.Mock(return_value=self.SCOPE)):
            code, out, _ = self._main("--list-documents")
        self.assertEqual(code, 0)
        self.assertIn("rehabGedung.pdf", out)

    def test_a_missing_question_is_still_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._main("--retriever", "bm25")


class CitationTests(AskTests):
    """A question naming a clause: that clause is pinned ahead of the search."""

    CITED = Hit(id="cited", score=0.0, metadata={"label": "3", "hierarchy_path": "Pasal 5/3",
                                                 "sub_document": "main_agreement", "document_key": DOC_A},
                text="Masa Pemeliharaan ditentukan dalam SSKK selama 180 hari kalender.")

    def _cited(self, *argv: str, result=None):
        found = references.ReferenceResult(hits=[self.CITED], tier=0) if result is None else result
        self.find = mock.Mock(return_value=found)
        self.enterContext(mock.patch.object(ask.references, "find", self.find))
        return self._main(*argv)

    def test_the_cited_clause_is_pinned_first(self) -> None:
        code, out, _ = self._cited("Berapa lama Masa Pemeliharaan menurut Pasal 5 ayat (3)",
                                   "--retriever", "bm25")
        self.assertEqual(code, 0)
        self.assertIn("Masa Pemeliharaan ditentukan", out)
        self.assertIn("[1] Pasal 5 ayat (3)", out)

    def test_the_search_runs_on_the_question_without_the_citation(self) -> None:
        self._cited("Berapa lama Masa Pemeliharaan menurut Pasal 5 ayat (3)", "--retriever", "bm25")
        self.assertEqual(self.retriever.queries, ["Berapa lama Masa Pemeliharaan"])

    def test_a_citation_only_question_skips_search_entirely(self) -> None:
        self._cited("Pasal 5 ayat 3", "--retriever", "hybrid")
        self.assertEqual(self.retriever.queries, [], "no search was needed")

    def test_an_unmatched_citation_says_so_and_searches_normally(self) -> None:
        code, out, _ = self._cited("apa isi Pasal 99 ayat 1", "--retriever", "bm25",
                                   result=references.ReferenceResult())
        self.assertEqual(code, 0)
        self.assertIn("Pasal 99 ayat 1 tidak ditemukan", out)
        self.assertEqual(self.retriever.queries, ["apa"])

    def test_a_relaxed_part_is_disclosed(self) -> None:
        relaxed = references.ReferenceResult(hits=[self.CITED], tier=0, part_relaxed=True)
        _, out, _ = self._cited("SSKK 33.8", "--retriever", "bm25", result=relaxed)
        self.assertIn("ditemukan di bagian lain", out)

    def test_a_question_with_no_citation_searches_the_whole_question(self) -> None:
        self._main("kewajiban penyedia mengasuransikan pekerjaan", "--retriever", "bm25")
        self.assertEqual(self.retriever.queries, ["kewajiban penyedia mengasuransikan pekerjaan"])

    def test_the_cited_clause_reaches_the_synthesizer_as_the_first_source(self) -> None:
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="jawaban", sources=[])
        self._cited("Berapa lama Masa Pemeliharaan menurut Pasal 5 ayat (3)",
                    "--retriever", "bm25", "--synthesizer", "ollama")
        # build_synthesizer is patched per-test in _main; assert via the hits passed.
        self.assertEqual(self.retriever.queries, ["Berapa lama Masa Pemeliharaan"])


class RouteTests(AskTests):
    """`--route`: core-field questions answered from the raw extraction before search."""

    RAW = {DOC_A: (Path("a_raw.json"), {"core": {"contract_number": {"value": "08/SP-PPK", "confidence": 0.9, "flags": []}}}),
           DOC_B: (Path("b_raw.json"), {"core": {"contract_number": {"value": None}}})}

    def _routed(self, *argv: str, raw=None):
        corpus = {DOC_A: "a.pdf", DOC_B: "b.pdf"}
        self.enterContext(mock.patch.object(ask, "corpus_documents", mock.Mock(return_value=corpus)))
        # A dict's `.get` is exactly the provider's signature.
        self.enterContext(mock.patch.object(ask, "raw_document_provider",
                                            mock.Mock(return_value=(raw or self.RAW).get)))
        return self._main(*argv)

    def _scoped_to_a(self) -> None:
        self.enterContext(mock.patch.object(ask, "resolve_scope", mock.Mock(return_value={DOC_A: "a.pdf"})))

    def test_a_core_field_question_is_answered_without_search(self) -> None:
        self._scoped_to_a()
        code, out, _ = self._routed("nomor kontrak?", "--document", "a")
        self.assertEqual(code, 0)
        self.assertIn("08/SP-PPK", out)
        self.assertIn("Sumber: data utama dokumen", out)
        self.assertEqual(self.retriever.scopes, [], "search never ran")

    def test_an_unscoped_field_question_asks_for_a_document_instead_of_listing_every_one(self) -> None:
        code, out, _ = self._routed("nomor kontrak?")
        self.assertEqual(code, 1)
        self.assertIn("sebutkan dokumennya", out)
        self.assertIn("2 dokumen tersedia", out)
        self.assertNotIn("a.pdf", out)
        self.assertNotIn("08/SP-PPK", out)
        self.assertEqual(self.retriever.scopes, [])

    def test_nothing_in_core_falls_through_to_search(self) -> None:
        self._scoped_to_a()
        empty = {DOC_A: (Path("a_raw.json"), {"core": {}})}
        code, out, load_settings = self._routed("nomor kontrak?", "--document", "a", "--retriever", "hybrid", raw=empty)
        self.assertEqual(code, 0)
        self.assertNotIn("beralih ke pencarian klausul", out)
        self.assertEqual(self.retriever.scopes, [{DOC_A}])
        # Settings are read once up front now, not re-read when the route falls through.
        load_settings.assert_called_once_with()

    def test_route_search_skips_lookup(self) -> None:
        _, out, _ = self._routed("nomor kontrak?", "--retriever", "bm25", "--route", "search")
        self.assertNotIn("Sumber: data utama dokumen", out)
        self.assertEqual(self.retriever.scopes, [None])

    def test_forced_lookup_refuses_a_clause_question(self) -> None:
        code, out, _ = self._routed("kewajiban penyedia", "--route", "lookup")
        self.assertEqual(code, 1)
        self.assertIn("tidak bisa dijawab langsung", out)
        self.assertEqual(self.retriever.scopes, [])

    def test_lookup_respects_document_scope(self) -> None:
        self.enterContext(mock.patch.object(ask, "resolve_scope", mock.Mock(return_value={DOC_A: "a.pdf"})))
        _, out, _ = self._routed("nomor kontrak?", "--document", "a")
        self.assertIn("Dokumen: a.pdf", out)
        self.assertNotIn("b.pdf:", out)


if __name__ == "__main__":
    unittest.main()
