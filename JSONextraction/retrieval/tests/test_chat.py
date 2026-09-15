"""Unit tests for retrieval.chat.

The Mistral client is faked throughout: no API key, no tokens, and the prompt
rules stay testable without depending on what the model says today — the same
discipline the retrieval tests follow.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrieval.chat import (
    MistralSynthesizer,
    NullSynthesizer,
    build_prompt,
    build_synthesizer,
    collapse_duplicates,
)
from retrieval.retrievers import Hit

RETRIEVAL_DIR = Path(__file__).resolve().parents[1]


def _hit(row_id: str, text: str, label: str = "55", document: str = "aaaaaaaa") -> Hit:
    return Hit(
        id=row_id,
        score=0.1,
        metadata={
            "label": label,
            "sub_document": "general_terms",
            "hierarchy_path": f"C/{label}",
            "document_key": document,
        },
        text=text,
    )


class FakeChatClient:
    def __init__(self, content: str = "Jawaban [Pasal 55].") -> None:
        self.content = content
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(complete=self._complete)

    def _complete(self, *, model, messages, temperature):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(total_tokens=123),
        )


def _synthesizer(client: FakeChatClient) -> MistralSynthesizer:
    synth = MistralSynthesizer.__new__(MistralSynthesizer)
    synth._client = client
    synth.model = "mistral-small-latest"
    synth.temperature = 0.0
    return synth


class LayerSeparationTests(unittest.TestCase):
    def test_retrieval_layer_does_not_import_chat(self) -> None:
        """The load-bearing architectural rule, asserted rather than trusted.

        Synthesis must be removable without touching retrieval, so the arrow
        points one way only: chat imports from the retrieval path, never the
        reverse. This mirrors how `ocr_main.py` and `main.py` both build on the
        shared stages without either importing the other.
        """
        offenders = []
        for path in RETRIEVAL_DIR.glob("*.py"):
            if path.name in ("chat.py", "ask.py"):
                continue  # the synthesis layer and the one wiring point
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("chat"):
                    offenders.append(path.name)
                elif isinstance(node, ast.Import):
                    if any(alias.name.endswith("chat") for alias in node.names):
                        offenders.append(path.name)
        self.assertEqual(offenders, [], "retrieval modules must not import the chat layer")

    def test_the_gate_runs_without_a_chat_model_configured(self) -> None:
        """`retrieval_evaluate` must not acquire a dependency on synthesis being
        set up — the regression gate has to run on a machine with no chat model
        at all."""
        from retrieval.config import Settings

        settings = Settings(
            api_key="k", model="m", batch_size=1, request_delay=0.0,
            db_path=Path("."), collection="c",
        )
        self.assertEqual(settings.chat_model, "")


class CollapseDuplicatesTests(unittest.TestCase):
    def test_identical_clauses_from_several_contracts_collapse_to_one(self) -> None:
        """The requirement that came out of measurement: 60% of the corpus is
        duplicate text, so an uncollapsed top-5 is often one sentence five
        times."""
        hits = [
            _hit("a", "Besarnya denda keterlambatan adalah 1/1000.", document="aaaaaaaa"),
            _hit("b", "Besarnya denda keterlambatan adalah 1/1000.", document="bbbbbbbb"),
            _hit("c", "Besarnya denda keterlambatan adalah 1/1000.", document="cccccccc"),
        ]
        sources = collapse_duplicates(hits)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].copies, 3)
        self.assertEqual(sorted(sources[0].documents), ["aaaaaaaa", "bbbbbbbb", "cccccccc"])

    def test_collapsing_keeps_the_best_ranked_copy(self) -> None:
        sources = collapse_duplicates([_hit("first", "same text"), _hit("second", "same text")])
        self.assertEqual(sources[0].id, "first")

    def test_line_wrapping_differences_still_collapse(self) -> None:
        """Copies differ in whitespace because the specimens are typeset
        differently; extraction preserves that faithfully. Byte equality alone
        would leave them uncollapsed."""
        sources = collapse_duplicates([_hit("a", "denda  keterlambatan\nadalah"), _hit("b", "denda keterlambatan adalah")])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].copies, 2)

    def test_different_clauses_are_not_merged(self) -> None:
        sources = collapse_duplicates([_hit("a", "tentang asuransi"), _hit("b", "tentang denda")])
        self.assertEqual(len(sources), 2)

    def test_empty_text_is_dropped_rather_than_offered_as_a_source(self) -> None:
        self.assertEqual(collapse_duplicates([_hit("a", "   ")]), [])


class PromptTests(unittest.TestCase):
    def test_prompt_carries_every_clause_and_its_citation(self) -> None:
        messages = build_prompt("berapa denda?", collapse_duplicates([_hit("a", "isi klausul", label="62")]))
        user = messages[1]["content"]

        self.assertIn("isi klausul", user)
        self.assertIn("Pasal 62", user)
        self.assertIn("berapa denda?", user)

    def test_prompt_forbids_outside_knowledge_and_guessing(self) -> None:
        """The extractive contract, mirroring the pipeline's own "never guess"
        rule. If these instructions are ever dropped, the layer stops being
        safe for legal text."""
        system = build_prompt("q", [])[0]["content"]
        self.assertIn("ONLY", system)
        self.assertIn("Never use outside knowledge", system)
        self.assertIn("Do not guess", system)

    def test_prompt_explains_blank_templates(self) -> None:
        """Placeholders are correct output, not missing data. The
        model has to be told, or it reports them as gaps."""
        self.assertIn("blank templates", build_prompt("q", [])[0]["content"])

    def test_duplicate_count_is_disclosed_to_the_model(self) -> None:
        """Otherwise three photocopies of one clause read as three independent
        sources agreeing."""
        sources = collapse_duplicates([_hit("a", "t", document="aa"), _hit("b", "t", document="bb")])
        self.assertIn("appears identically in 2", build_prompt("q", sources)[1]["content"])


class SynthesizerTests(unittest.TestCase):
    def test_null_synthesizer_makes_no_model_call_and_returns_the_clauses(self) -> None:
        answer = NullSynthesizer().synthesize("q", [_hit("a", "isi klausul", label="55")])
        self.assertIn("isi klausul", answer.text)
        self.assertIn("Pasal 55", answer.text)
        self.assertIsNone(answer.model)

    def test_mistral_synthesizer_returns_content_sources_and_usage(self) -> None:
        client = FakeChatClient("Penyedia wajib mengasuransikan [Pasal 55].")
        answer = _synthesizer(client).synthesize("kewajiban asuransi?", [_hit("a", "isi klausul")])

        self.assertEqual(answer.text, "Penyedia wajib mengasuransikan [Pasal 55].")
        self.assertEqual(answer.usage_tokens, 123)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(client.calls[0]["temperature"], 0.0)

    def test_no_model_call_when_nothing_was_retrieved(self) -> None:
        """With no clauses there is nothing to ground an answer in, and asking
        anyway invites exactly the unsourced response the prompt forbids."""
        client = FakeChatClient()
        answer = _synthesizer(client).synthesize("q", [])

        self.assertEqual(client.calls, [])
        self.assertEqual(answer.sources, [])
        self.assertIn("Tidak ada klausul", answer.text)

    def test_sources_are_the_collapsed_set_not_the_raw_hits(self) -> None:
        client = FakeChatClient()
        hits = [_hit("a", "same", document="aa"), _hit("b", "same", document="bb")]
        answer = _synthesizer(client).synthesize("q", hits)

        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].copies, 2)

    def test_unpinned_chat_model_is_refused(self) -> None:
        """An answer that cannot be attributed to a known model version is not
        reproducible — the same rule EMBEDDING_MODEL follows."""
        with self.assertRaises(SystemExit):
            MistralSynthesizer("key", "")

    def test_build_synthesizer_names(self) -> None:
        from retrieval.config import Settings

        settings = Settings(
            api_key="k", model="m", batch_size=1, request_delay=0.0,
            db_path=Path("."), collection="c", chat_model="mistral-small-latest",
        )
        self.assertEqual(build_synthesizer("null", settings).name, "null")
        with self.assertRaises(ValueError):
            build_synthesizer("gpt", settings)

    def test_every_synthesizer_satisfies_the_interface(self) -> None:
        """`Synthesizer` is a Protocol nobody inherits from, so conformance is
        structural — and asserted here, so a new implementation that drifts
        from the shape `ask.py` calls fails a test instead of a live run."""
        from unittest import mock

        from retrieval.chat import Synthesizer

        with mock.patch("mistralai.client.Mistral"):
            implementations = [NullSynthesizer(), MistralSynthesizer("key", "open-mistral-nemo")]
        for implementation in implementations:
            with self.subTest(name=implementation.name):
                self.assertIsInstance(implementation, Synthesizer)
        self.assertNotIsInstance(object(), Synthesizer)


if __name__ == "__main__":
    unittest.main()
