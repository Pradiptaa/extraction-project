"""Mistral embedding calls with retry/backoff and structured failure logging.

Thin on purpose: this module knows how to turn a list of strings into a list of
vectors and nothing about Chroma, checkpoints, or the embedding view. The batch
job in `load.py` owns all of that.

Import note: `mistralai` 2.x is a namespace package with no top-level
`__init__.py`, so the widely-documented `from mistralai import Mistral` raises
ImportError. The real path is `mistralai.client`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# 429 is the free-tier per-second cap; 5xx are transient server faults. Anything
# else (401 bad key, 422 malformed input) is a bug or a config error and must
# fail immediately — retrying a bad key just burns five attempts and hides the
# real message.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, SDKError):
        response = getattr(exc, "raw_response", None)
        return getattr(response, "status_code", None) in _RETRYABLE_STATUS
    # Connection resets / timeouts surface as httpx errors, which are worth
    # another attempt.
    return isinstance(exc, (TimeoutError, ConnectionError))


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    total_tokens: int
    dimension: int


class Embedder:
    """Wraps the Mistral client and enforces one invariant: every vector this
    returns has the same width as the first one it ever saw.

    A silent width change mid-run would poison the Chroma collection — Chroma
    rejects a mismatched vector on write, so the job would die partway with a
    confusing error rather than at the point the model actually changed.
    """

    def __init__(self, api_key: str, model: str, request_delay: float = 0.0) -> None:
        self._client = Mistral(api_key=api_key)
        self.model = model
        self.request_delay = request_delay
        self.dimension: int | None = None
        self.total_tokens = 0

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call(self, texts: list[str]) -> EmbeddingResult:
        response = self._client.embeddings.create(model=self.model, inputs=texts)
        vectors = [item.embedding for item in response.data]
        return EmbeddingResult(
            vectors=vectors,
            total_tokens=response.usage.total_tokens,
            dimension=len(vectors[0]) if vectors else 0,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.request_delay:
            time.sleep(self.request_delay)

        result = self._call(texts)

        if len(result.vectors) != len(texts):
            raise RuntimeError(
                f"asked for {len(texts)} embeddings, got {len(result.vectors)} — "
                "refusing to guess which input each vector belongs to"
            )

        if self.dimension is None:
            self.dimension = result.dimension
            logger.info("embedding dimension: %d (model=%s)", self.dimension, self.model)
        elif result.dimension != self.dimension:
            raise RuntimeError(
                f"embedding dimension changed mid-run: {self.dimension} -> {result.dimension}. "
                "The model likely changed underneath us; stopping before this corrupts the collection."
            )

        self.total_tokens += result.total_tokens
        return result.vectors
