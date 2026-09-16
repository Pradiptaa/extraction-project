"""Answer synthesis over retrieved clauses — a layer ABOVE retrieval.

The dependency arrow points one way and must stay that way: this module imports
from the retrieval path, and **nothing in the retrieval path imports this**.
`retrieval_evaluate.py`, `retrievers.py`, `load.py` and `embed.py` all run with
no chat model configured and no knowledge that synthesis exists. That is the
same rule `ocr_main.py` and `main.py` follow around the shared stages, and it is
enforced by a test (`test_chat.test_retrieval_layer_does_not_import_chat`).

Two consequences worth stating, because they are the point:

- **The gate never scores generated prose.** `retrieval_evaluate.py` measures
  whether the right clause comes back, which is checkable against ground truth.
  Whether a model wrote a good paragraph from it is not, and mixing the two
  would turn a regression gate into a vibe check.
- **Synthesis can be swapped or removed** — a different provider, a local
  model, or `NullSynthesizer` — without touching retrieval logic.

The prompt is deliberately extractive. This is contract law: a plausible
sentence that is not in the source is worse than no answer, so the model is told
to answer only from the supplied clauses, to cite them, and to say when they do
not contain the answer. That mirrors the extraction pipeline's own contract —
it never guesses, and an unresolved field is a documented null.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .embed import is_retryable
from .retrievers import Hit

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")

# Short names a reader of these contracts would recognise. A citation of
# "special_terms" or a positional table id means nothing to anyone; "SSKK" is
# what the document calls itself.
SUB_DOCUMENT_LABELS = {
    "main_agreement": "Surat Perjanjian",
    "general_terms": "SSUK",
    "special_terms": "SSKK",
    "annex_a": "Lampiran A",
    "annex_b": "Lampiran B",
}


def parse_ref_targets(raw: str) -> list[tuple[str, str]]:
    """`"general_terms:B/27/27.1;general_terms:A/4/4.2"` -> [(sub_doc, clause)].

    Chroma metadata has to be a flat string, so `refs` is flattened on load and
    parsed back here. Only the last path segment is kept: `B/27/27.1` is the
    section letter, the clause and the sub-clause, and a citation wants the
    sub-clause a reader would look up.
    """
    targets: list[tuple[str, str]] = []
    for part in (raw or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sub_document, _, path = part.partition(":")
        clause = path.rsplit("/", 1)[-1].strip()
        if clause and clause != "raw":
            pair = (sub_document.strip(), clause)
            if pair not in targets:  # the same target is often cited twice in one row
                targets.append(pair)
    return targets


SYSTEM_PROMPT = """\
You answer questions about Indonesian government construction contracts \
(Perpres 16/2018 standard form) using ONLY the contract clauses provided.

Rules:
- Answer only from the supplied clauses. Never use outside knowledge of \
Indonesian law, and never infer a provision that is not written.
- Cite the clause you used for each statement, e.g. [Pasal 55.2].
- Use ONLY the identifier printed in a clause's header, copied exactly. Never \
build a citation out of numbers found inside the clause text — a row beginning \
"27.1 | ..." is not necessarily clause 27.1. If a header says a row refers to \
another clause ("mengacu SSUK 27.1"), you may say so, but cite the header you \
were given.
- If the supplied clauses do not answer the question, say so plainly and stop. \
Do not guess, and do not offer the closest-sounding clause as if it answered.
- Some clauses are blank templates (e.g. "........ [diisi nama paket]"). \
That is the source document, not missing data — report the field as unfilled \
rather than inventing a value.
- Answer in the language of the question.\
"""


@dataclass
class SourceClause:
    """One clause handed to the model, plus how many identical copies the corpus
    held. `copies` is not decoration: it is the difference between "three
    contracts agree on this" and "this was retrieved three times"."""

    id: str
    text: str
    label: str
    sub_document: str
    hierarchy_path: str
    copies: int = 1
    documents: list[str] = field(default_factory=list)
    node_type: str = ""
    page: str = ""
    # Resolved cross-references as (sub_document, clause) — for a table row this
    # is the SSUK clause it is keyed to, which is the only identifier that means
    # anything to a reader.
    refs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def sub_document_label(self) -> str:
        return SUB_DOCUMENT_LABELS.get(self.sub_document, self.sub_document)

    @property
    def citation(self) -> str:
        """An identifier a reader can actually look up.

        A clause cites as `Pasal 55.2`. A **table row** has no label — its path
        is a positional table id (`t_062_0/3`) that is meaningless outside this
        codebase — so it cites by what it is and what it keys to:
        `SSKK hal. 62 (mengacu SSUK 27.1)`.

        This is not cosmetic. Handed `t_062_0/3`, a model does not print it; it
        reaches into the row's own text for something that looks like a clause
        number and cites that instead. Observed live: a row reading
        `27.1 | Masa Pelaksanaan | ...` was cited as `[27.1]` — correct by luck,
        because the leading cell happened to be the cross-reference. On a row
        whose first cell is a price or a date the same behaviour invents a
        citation with full confidence. Giving every source a meaningful,
        checkable identifier removes the reason to guess.
        """
        if self.label:
            # An ayat's label is a bare ordinal ("2"), so "Pasal 2" would name a
            # different provision entirely — the Surat Perjanjian has both a
            # Pasal 2 and a Pasal 5 ayat (2). Observed: the ayat holding the
            # Masa Pelaksanaan was cited as "Pasal 2", and the model then
            # reported that no Pasal 5 ayat (2) had been supplied — while
            # holding its text. The path carries the parent, so use it.
            parent, _, child = self.hierarchy_path.rpartition("/")
            if parent and child == self.label and parent.lower().startswith("pasal "):
                return f"{parent} ayat ({self.label})"
            return f"Pasal {self.label}"

        where = self.sub_document_label or self.hierarchy_path or self.id[:8]
        if self.page:
            where = f"{where} hal. {self.page}"
        if self.refs:
            cited = ", ".join(
                f"{SUB_DOCUMENT_LABELS.get(sub, sub)} {clause}" for sub, clause in self.refs
            )
            return f"{where} (mengacu {cited})"
        return where


@dataclass
class Answer:
    text: str
    sources: list[SourceClause]
    model: str | None = None
    usage_tokens: int = 0


def collapse_duplicates(hits: list[Hit]) -> list[SourceClause]:
    """Merge hits whose text is identical, keeping the best-ranked one.

    Not an optimisation — a correctness measure for the prompt. 60% of this
    corpus is duplicate text (all six specimens are the same standard form), so
    a top-5 is routinely the same sentence five times. Passing that to the model
    wastes the context window and, worse, makes the most-duplicated clause look
    like the most corroborated one when it is just the most photocopied.

    Collapsing here is safe in a way that corpus-wide dedup of the stored
    corpus is not: nothing is written, no id or metadata is destroyed, and
    the retrieval layer above still saw every row. Only the prompt is affected.

    Comparison is on whitespace-normalised text. Byte equality alone would miss
    copies that differ only in line-wrapping, which the extraction preserves
    faithfully from differently-typeset PDFs.
    """
    collapsed: dict[str, SourceClause] = {}
    for hit in hits:
        key = _WS_RE.sub(" ", hit.text or "").strip().lower()
        if not key:
            continue
        metadata = hit.metadata or {}
        document = str(metadata.get("document_key") or "")[:8]
        existing = collapsed.get(key)
        if existing is None:
            collapsed[key] = SourceClause(
                id=hit.id,
                text=hit.text,
                label=str(metadata.get("label") or ""),
                sub_document=str(metadata.get("sub_document") or ""),
                hierarchy_path=str(metadata.get("hierarchy_path") or ""),
                copies=1,
                documents=[document] if document else [],
                node_type=str(metadata.get("node_type") or ""),
                page=str(metadata.get("page_first") or ""),
                refs=parse_ref_targets(str(metadata.get("ref_targets") or "")),
            )
        else:
            existing.copies += 1
            if document and document not in existing.documents:
                existing.documents.append(document)
    return list(collapsed.values())


def build_prompt(question: str, sources: list[SourceClause], scope_note: str = "") -> list[dict]:
    """`scope_note` names the single contract the clauses came from, when the
    caller restricted retrieval to one.

    It must reach the model. Without it the prompt looks identical to a
    corpus-wide one, so an answer drawn from a single contract reads as though
    it describes the contracts in general — the model has no other way to know
    the difference, and a confident over-generalisation about legal text is
    exactly what the extractive rules exist to prevent.
    """
    blocks = []
    for n, source in enumerate(sources, start=1):
        header = f"[{n}] {source.citation}"
        # A label-based citation ("Pasal 55.2") does not say which part of the
        # contract it is in; a table row's already does, so appending it again
        # would read as two different locations.
        if source.sub_document and not source.citation.startswith(source.sub_document_label):
            header += f" ({source.sub_document_label})"
        if source.copies > 1:
            # Told to the model explicitly: identical text across contracts is
            # one provision of a standard form, not corroborating evidence.
            header += f" — appears identically in {source.copies} retrieved rows"
        blocks.append(f"{header}\n{source.text}")

    body = "\n\n".join(blocks) if blocks else "(no clauses were retrieved)"
    preamble = ""
    if scope_note:
        preamble = (
            f"Semua klausul di bawah berasal HANYA dari satu kontrak: {scope_note}. "
            "Jawab tentang kontrak itu saja, dan jangan menyatakan apa pun tentang "
            "kontrak lain.\n\n"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{preamble}Klausul kontrak:\n\n{body}\n\nPertanyaan: {question}",
        },
    ]


@runtime_checkable
class Synthesizer(Protocol):
    """Interface. `synthesize` turns a question plus retrieved hits into an
    Answer. Implementations must not retrieve anything themselves — what to
    retrieve is the caller's decision, which is what keeps the two layers
    swappable.

    A Protocol, like `retrievers.Retriever`, rather than a base class nobody
    inherited from: the implementations below satisfy it structurally, and a
    test asserts they do, so a new synthesizer cannot silently drift from
    the shape `ask.py` calls."""

    name: str

    def synthesize(self, question: str, hits: list[Hit], scope_note: str = "") -> Answer:
        """An Answer grounded only in `hits`.

        `scope_note` names the single contract `hits` were restricted to, if
        any. Optional so that adding scoping did not change how any existing
        caller invokes a synthesizer.
        """


class NullSynthesizer:
    """Returns the retrieved clauses with no model call at all.

    The default, and not a stub: it is how you read raw retrieval output, how
    the CLI runs with no API budget, and the control case when judging whether
    synthesis is adding anything or just paraphrasing.
    """

    name = "null"

    def synthesize(self, question: str, hits: list[Hit], scope_note: str = "") -> Answer:
        sources = collapse_duplicates(hits)
        lines = [f"[{n}] {s.citation}: {s.text}" for n, s in enumerate(sources, start=1)]
        return Answer(text="\n".join(lines) or "(nothing retrieved)", sources=sources)


class MistralSynthesizer:
    """Mistral chat completion over the retrieved clauses.

    Shares `embed.is_retryable` rather than defining its own rule, so a 429 is
    retried and a 401 fails immediately on both endpoints — one policy, not two
    that drift apart.
    """

    name = "mistral"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0) -> None:
        from mistralai.client import Mistral

        if not model:
            raise SystemExit(
                "CHAT_MODEL is not set. Pin it in retrieval/.env — an unpinned chat model "
                "means an answer cannot be attributed to a known model version."
            )
        self._client = Mistral(api_key=api_key)
        self.model = model
        # Temperature 0 by default: this is a quoting task over legal text, and
        # sampling variety here shows up as invented paraphrase.
        self.temperature = temperature

    @retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call(self, messages: list[dict]):
        return self._client.chat.complete(
            model=self.model, messages=messages, temperature=self.temperature
        )

    def synthesize(self, question: str, hits: list[Hit], scope_note: str = "") -> Answer:
        sources = collapse_duplicates(hits)
        if not sources:
            # No model call: with nothing retrieved there is nothing to ground
            # an answer in, and asking anyway invites exactly the unsourced
            # response the prompt forbids.
            logger.warning("nothing retrieved for %r — not calling the model", question)
            return Answer(
                text="Tidak ada klausul yang ditemukan untuk pertanyaan ini.",
                sources=[],
                model=self.model,
            )

        logger.info(
            "synthesising over %d clauses (collapsed from %d hits)", len(sources), len(hits)
        )
        response = self._call(build_prompt(question, sources, scope_note))
        usage = getattr(response, "usage", None)

        return Answer(
            text=response.choices[0].message.content,
            sources=sources,
            model=self.model,
            usage_tokens=getattr(usage, "total_tokens", 0) or 0,
        )


def build_synthesizer(name: str, settings) -> Synthesizer:
    if name == "null":
        return NullSynthesizer()
    if name == "mistral":
        return MistralSynthesizer(settings.api_key, settings.chat_model)
    raise ValueError(f"unknown synthesizer {name!r} (expected null or mistral)")
