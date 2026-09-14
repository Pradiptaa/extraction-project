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

SYSTEM_PROMPT = """\
You answer questions about Indonesian government construction contracts \
(Perpres 16/2018 standard form) using ONLY the contract clauses provided.

Rules:
- Answer only from the supplied clauses. Never use outside knowledge of \
Indonesian law, and never infer a provision that is not written.
- Cite the clause you used for each statement, e.g. [Pasal 55.2].
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

    @property
    def citation(self) -> str:
        return f"Pasal {self.label}" if self.label else (self.hierarchy_path or self.id[:8])


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
            )
        else:
            existing.copies += 1
            if document and document not in existing.documents:
                existing.documents.append(document)
    return list(collapsed.values())


def build_prompt(question: str, sources: list[SourceClause]) -> list[dict]:
    blocks = []
    for n, source in enumerate(sources, start=1):
        header = f"[{n}] {source.citation}"
        if source.sub_document:
            header += f" ({source.sub_document})"
        if source.copies > 1:
            # Told to the model explicitly: identical text across contracts is
            # one provision of a standard form, not corroborating evidence.
            header += f" — appears identically in {source.copies} retrieved rows"
        blocks.append(f"{header}\n{source.text}")

    body = "\n\n".join(blocks) if blocks else "(no clauses were retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Klausul kontrak:\n\n{body}\n\nPertanyaan: {question}"},
    ]


class Synthesizer:
    """Interface. `synthesize` turns a question plus retrieved hits into an
    Answer. Implementations must not retrieve anything themselves — what to
    retrieve is the caller's decision, which is what keeps the two layers
    swappable."""

    name = "synthesizer"

    def synthesize(self, question: str, hits: list[Hit]) -> Answer:  # pragma: no cover - interface
        raise NotImplementedError


class NullSynthesizer:
    """Returns the retrieved clauses with no model call at all.

    The default, and not a stub: it is how you read raw retrieval output, how
    the CLI runs with no API budget, and the control case when judging whether
    synthesis is adding anything or just paraphrasing.
    """

    name = "null"

    def synthesize(self, question: str, hits: list[Hit]) -> Answer:
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

    def synthesize(self, question: str, hits: list[Hit]) -> Answer:
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
        response = self._call(build_prompt(question, sources))
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
