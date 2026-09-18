"""Answer synthesis over retrieved clauses — a layer ABOVE retrieval.

The dependency arrow points one way: this imports from the retrieval path and
nothing there imports this, enforced by
`test_chat.test_retrieval_layer_does_not_import_chat`. So the gate never scores
generated prose, and synthesis can be swapped or removed.

The prompt is deliberately extractive: in contract law a plausible sentence that
is not in the source is worse than no answer.
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

# What the documents call themselves; "special_terms" means nothing to a reader.
SUB_DOCUMENT_LABELS = {
    "main_agreement": "Surat Perjanjian",
    "general_terms": "SSUK",
    "special_terms": "SSKK",
    "annex_a": "Lampiran A",
    "annex_b": "Lampiran B",
}


def parse_ref_targets(raw: str) -> list[tuple[str, str]]:
    """`"general_terms:B/27/27.1;general_terms:A/4/4.2"` -> [(sub_doc, clause)].

    Chroma metadata must be flat, so `refs` is flattened on load and parsed back
    here. Only the last path segment is kept — the sub-clause a reader looks up.
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
            if pair not in targets:  # one row often cites the same target twice
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
    held — the difference between three contracts agreeing and three retrievals."""

    id: str
    text: str
    label: str
    sub_document: str
    hierarchy_path: str
    copies: int = 1
    documents: list[str] = field(default_factory=list)
    node_type: str = ""
    page: str = ""
    # Resolved cross-references as (sub_document, clause).
    refs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def sub_document_label(self) -> str:
        return SUB_DOCUMENT_LABELS.get(self.sub_document, self.sub_document)

    @property
    def citation(self) -> str:
        """An identifier a reader can actually look up.

        A clause cites as `Pasal 55.2`. A table row has no label — its path is a
        positional table id meaningless outside this codebase — so it cites by
        what it is and what it keys to: `SSKK hal. 62 (mengacu SSUK 27.1)`.
        Without that, a model reaches into the row's own text for something that
        looks like a clause number and cites that instead.
        """
        if self.label:
            # An ayat's label is a bare ordinal, so "Pasal 2" would name a
            # different provision. The path carries the parent, so use it.
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

    A correctness measure, not an optimisation: much of this corpus is duplicate
    text, so a top-5 is routinely the same sentence five times, which makes the
    most-photocopied clause look like the most corroborated one. Nothing is
    written — only the prompt is affected. Comparison is whitespace-normalised,
    since copies differ in line-wrapping between differently-typeset PDFs.
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
    caller restricted retrieval to one. It must reach the model: otherwise a
    single-contract answer reads as a general one."""
    blocks = []
    for n, source in enumerate(sources, start=1):
        header = f"[{n}] {source.citation}"
        # A table row's citation already names its section; appending it again
        # would read as two different locations.
        if source.sub_document and not source.citation.startswith(source.sub_document_label):
            header += f" ({source.sub_document_label})"
        if source.copies > 1:
            # Stated explicitly so the model doesn't read repetition as corroboration.
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
    """Turns a question plus retrieved hits into an Answer. Implementations must
    not retrieve anything themselves — that is the caller's decision, which is
    what keeps the two layers swappable."""

    name: str

    def synthesize(self, question: str, hits: list[Hit], scope_note: str = "") -> Answer:
        """An Answer grounded only in `hits`. `scope_note` names the single
        contract they were restricted to, if any."""


class NullSynthesizer:
    """Returns the retrieved clauses with no model call. The default, and not a
    stub: it is how the CLI runs with no API budget, and the control case for
    judging whether synthesis adds anything."""

    name = "null"

    def synthesize(self, question: str, hits: list[Hit], scope_note: str = "") -> Answer:
        sources = collapse_duplicates(hits)
        lines = [f"[{n}] {s.citation}: {s.text}" for n, s in enumerate(sources, start=1)]
        return Answer(text="\n".join(lines) or "(nothing retrieved)", sources=sources)


class MistralSynthesizer:
    """Mistral chat completion over the retrieved clauses. Shares
    `embed.is_retryable` so both endpoints follow one retry policy."""

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
        # 0 by default: on a quoting task, sampling variety is invented paraphrase.
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
            # No model call: nothing retrieved means nothing to ground an answer in.
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
