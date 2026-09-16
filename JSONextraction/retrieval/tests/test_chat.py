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
    parse_ref_targets,
)
from retrieval.retrievers import Hit

RETRIEVAL_DIR = Path(__file__).resolve().parents[1]


def _table_row_hit(text: str, ref_targets: str = "general_terms:B/27/27.1", page: int = 62) -> Hit:
    """A real SSKK table row, shaped exactly as `load.py` writes it to Chroma:
    no label, a positional table id for a path, and its cross-references
    flattened into a string."""
    return Hit(
        id="ca8a094e49314357",
        score=0.1,
        metadata={
            "label": "",
            "sub_document": "special_terms",
            "hierarchy_path": "t_062_0/3",
            "document_key": "aaaaaaaa",
            "node_type": "table_row",
            "table_id": "t_062_0",
            "page_first": page,
            "ref_targets": ref_targets,
        },
        text=text,
    )


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

    def test_a_scoped_prompt_says_the_clauses_are_from_one_contract(self) -> None:
        """Pinned with the other prompt rules, for the same reason: the model
        cannot otherwise tell a single-contract question from a corpus-wide
        one, and would generalise a provision from one contract to all six."""
        content = build_prompt("q", [], scope_note="rehabGedung.pdf")[1]["content"]
        self.assertIn("rehabGedung.pdf", content)
        self.assertIn("HANYA dari satu kontrak", content)

    def test_an_unscoped_prompt_makes_no_claim_about_scope(self) -> None:
        self.assertNotIn("satu kontrak", build_prompt("q", [])[1]["content"])


class CitationTests(unittest.TestCase):
    """Every source must reach the model with an identifier a reader could look
    up, because the alternative is not "no citation" — it is an invented one.

    Observed live before this: a table row whose path is the positional
    `t_062_0/3` was cited by the model as `[27.1]`, a number it took from the
    row's own leading cell. Right by luck there; on a row whose first cell is a
    price or a date the same behaviour produces a confident, wrong citation of
    legal text.
    """

    def test_a_clause_cites_by_its_label(self) -> None:
        self.assertEqual(collapse_duplicates([_hit("a", "isi", label="55.2")])[0].citation, "Pasal 55.2")

    def test_an_ayat_is_cited_with_its_article(self) -> None:
        """An ayat's label is a bare ordinal, so "Pasal 2" would name a
        different provision — the Surat Perjanjian has both a Pasal 2 and a
        Pasal 5 ayat (2). Observed before this: the ayat holding the Masa
        Pelaksanaan was cited as "Pasal 2", and the model reported that no
        Pasal 5 ayat (2) had been supplied while holding its text."""
        hit = Hit(
            id="x", score=0.1, text="Masa Pelaksanaan ditentukan dalam SSKK",
            metadata={"label": "2", "hierarchy_path": "Pasal 5/2", "node_type": "subclause",
                      "sub_document": "main_agreement", "page_first": 4, "document_key": "aaaaaaaa"},
        )
        self.assertEqual(collapse_duplicates([hit])[0].citation, "Pasal 5 ayat (2)")

    def test_an_ssuk_subclause_keeps_its_dotted_label(self) -> None:
        """Only a bare-ordinal child of a Pasal is an ayat; 55.2 already names
        itself unambiguously and must not be rewritten."""
        self.assertEqual(collapse_duplicates([_hit("a", "isi", label="55.2")])[0].citation, "Pasal 55.2")

    def test_a_table_row_cites_by_what_it_is_and_what_it_refers_to(self) -> None:
        source = collapse_duplicates([_table_row_hit("27.1 | Masa Pelaksanaan | 120 hari")])[0]
        self.assertEqual(source.citation, "SSKK hal. 62 (mengacu SSUK 27.1)")

    def test_a_table_row_citation_never_shows_the_positional_table_id(self) -> None:
        """`t_062_0/3` identifies nothing outside this codebase, and handing it
        to the model is what made it look for a number elsewhere."""
        source = collapse_duplicates([_table_row_hit("27.1 | Masa Pelaksanaan | 120 hari")])[0]
        self.assertNotIn("t_062_0", source.citation)

    def test_several_cross_references_are_all_named(self) -> None:
        source = collapse_duplicates(
            [_table_row_hit("Korespondensi | ...", ref_targets="general_terms:A/4/4.1;general_terms:A/4/4.2")]
        )[0]
        self.assertEqual(source.citation, "SSKK hal. 62 (mengacu SSUK 4.1, SSUK 4.2)")

    def test_a_row_with_no_label_and_no_refs_still_cites_something_locatable(self) -> None:
        source = collapse_duplicates([_table_row_hit("Nilai | Rp0,00", ref_targets="")])[0]
        self.assertEqual(source.citation, "SSKK hal. 62")

    def test_the_prompt_forbids_citing_numbers_from_the_clause_body(self) -> None:
        """Pinned with the other extractive rules. Without it the model has no
        reason not to read a citation out of the text it was given."""
        system = build_prompt("q", [])[0]["content"]
        self.assertIn("ONLY the identifier printed in a clause's header", system)
        self.assertIn("Never build a citation out of numbers found inside the clause text", system)

    def test_the_sub_document_is_not_repeated_when_the_citation_holds_it(self) -> None:
        source = collapse_duplicates([_table_row_hit("27.1 | Masa Pelaksanaan | 120 hari")])[0]
        header = build_prompt("q", [source])[1]["content"].splitlines()[2]
        self.assertEqual(header, "[1] SSKK hal. 62 (mengacu SSUK 27.1)")
        self.assertNotIn("special_terms", header)

    def test_a_clause_header_still_names_its_part_of_the_contract(self) -> None:
        source = collapse_duplicates([_hit("a", "isi", label="55.2")])[0]
        self.assertIn("[1] Pasal 55.2 (SSUK)", build_prompt("q", [source])[1]["content"])


class RefTargetParsingTests(unittest.TestCase):
    def test_the_last_path_segment_is_the_clause_a_reader_looks_up(self) -> None:
        self.assertEqual(parse_ref_targets("general_terms:B/27/27.1"), [("general_terms", "27.1")])

    def test_repeated_targets_are_collapsed(self) -> None:
        """A single row often cites the same clause twice; naming it twice in a
        citation reads as two separate references."""
        self.assertEqual(
            parse_ref_targets("general_terms:A/6/6.3;general_terms:A/6/6.3"),
            [("general_terms", "6.3")],
        )

    def test_unresolved_references_are_dropped(self) -> None:
        """`?:raw` means extraction could not resolve it — citing it would
        assert a link the source does not support."""
        self.assertEqual(parse_ref_targets("?:raw"), [])

    def test_empty_and_malformed_input_is_not_an_error(self) -> None:
        for raw in ("", "   ", ";;", "no_colon_here"):
            self.assertEqual(parse_ref_targets(raw), [], raw)


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
