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

from retrieval import ask
from retrieval.chat import Answer, SourceClause
from retrieval.config import Settings
from retrieval.retrievers import Hit

DOC_A = "aaaa1111"
DOC_B = "bbbb2222"

HITS = [
    Hit(id="a", score=0.1, metadata={"label": "55.2", "sub_document": "general_terms", "document_key": DOC_A}, text="Penyedia wajib mengasuransikan."),
    Hit(id="b", score=0.1, metadata={"label": "55.2", "sub_document": "general_terms", "document_key": DOC_B}, text="Penyedia wajib mengasuransikan."),
]
SETTINGS = Settings(api_key="", model="mistral-embed", batch_size=1, request_delay=0.0,
                    db_path=Path("."), collection="c", chat_model="open-mistral-nemo")


class FakeRetriever:
    def __init__(self) -> None:
        self.scopes: list[set[str] | None] = []

    def search(self, query: str, k: int, scope: set[str] | None = None) -> list[Hit]:
        self.scopes.append(scope)
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

    def test_bm25_with_null_synthesizer_does_not_require_an_api_key(self) -> None:
        _, _, load_settings = self._main("asuransi", "--retriever", "bm25")
        load_settings.assert_called_once_with(require_api_key=False)

    def test_dense_retrieval_or_mistral_synthesis_requires_an_api_key(self) -> None:
        _, _, load_settings = self._main("asuransi", "--retriever", "dense")
        load_settings.assert_called_once_with(require_api_key=True)

    def test_synthesis_failure_falls_back_to_the_retrieved_clauses(self) -> None:
        """Retrieval already succeeded; a chat outage must not throw that away."""
        failing = mock.Mock()
        failing.synthesize.side_effect = RuntimeError("429 rate limited")
        with self.assertLogs("retrieval.ask", level="ERROR") as captured:
            code, out, _ = self._main("asuransi", "--synthesizer", "mistral", synthesizer=failing)
        self.assertEqual(code, 1)
        self.assertIn("Penyedia wajib mengasuransikan.", out)
        self.assertIn("synthesis unavailable", out)
        self.assertIn("falling back", "\n".join(captured.output))

    def test_synthesized_answer_lists_its_sources_with_copy_count(self) -> None:
        source = SourceClause(id="a", text="Penyedia wajib mengasuransikan.", label="55.2",
                              sub_document="general_terms", hierarchy_path="C/55/55.2", copies=2)
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="Menurut [Pasal 55.2] ...", sources=[source],
                                                     model="open-mistral-nemo", usage_tokens=314)
        code, out, _ = self._main("asuransi", "--synthesizer", "mistral", synthesizer=synthesizer)
        self.assertEqual(code, 0)
        self.assertIn("Menurut [Pasal 55.2]", out)
        self.assertIn("Sumber:", out)
        self.assertIn("(x2 identik)", out)
        self.assertIn("(314 tokens)", out)


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
        self.assertIn("scope: rehabGedung.pdf", out)

    def test_an_empty_scoped_result_says_the_scope_caused_it(self) -> None:
        _, out, _ = self._main("asuransi", "--retriever", "bm25", "--document", "x",
                               scope={"no_such_document": "ghost.pdf"})
        self.assertIn("nothing matched in ghost.pdf", out)

    def test_the_synthesizer_is_told_which_contract_the_clauses_came_from(self) -> None:
        """Otherwise an answer about one contract reads as a claim about all."""
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="jawaban", sources=[])
        self._main("asuransi", "--retriever", "bm25", "--synthesizer", "mistral",
                   "--document", "rehab", synthesizer=synthesizer, scope=self.SCOPE)
        self.assertEqual(synthesizer.synthesize.call_args.args[2], "rehabGedung.pdf")

    def test_an_unscoped_run_passes_no_scope_note(self) -> None:
        synthesizer = mock.Mock()
        synthesizer.synthesize.return_value = Answer(text="jawaban", sources=[])
        self._main("asuransi", "--retriever", "bm25", "--synthesizer", "mistral", synthesizer=synthesizer)
        self.assertEqual(synthesizer.synthesize.call_args.args[2], "")

    def test_listing_documents_needs_neither_a_question_nor_a_key(self) -> None:
        with mock.patch.object(ask, "corpus_documents", mock.Mock(return_value=self.SCOPE)):
            code, out, load_settings = self._main("--list-documents")
        self.assertEqual(code, 0)
        self.assertIn("rehabGedung.pdf", out)
        load_settings.assert_called_once_with(require_api_key=False)

    def test_a_missing_question_is_still_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._main("--retriever", "bm25")


if __name__ == "__main__":
    unittest.main()
