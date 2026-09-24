"""Embedding calls against a local Ollama, with retry/backoff. Knows nothing
about Chroma or checkpoints — `load.py` owns that.

Local serving changes what can go wrong, not what must be guaranteed. There is
no rate limit and no bill, so pacing is gone; but the server can be down, the
model unpulled, and a cold model takes tens of seconds to load. The invariants
that protect the collection — every vector the same width, one vector per input
— are unchanged, because a mis-shaped batch corrupts the store just as badly
whoever served it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .config import DEFAULT_OLLAMA_HOST

logger = logging.getLogger(__name__)

# A cold model is loaded on the first request, which on a small GPU is tens of
# seconds. Timing that out would retry the load from scratch, and never finish.
DEFAULT_TIMEOUT = 300.0

# Everything else (404 for an unpulled model, 400 for a malformed request) is a
# config error and must fail immediately.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    """Public because `chat.py` reuses this exact policy."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    # Ollama restarting, or not up yet, is worth waiting out. A bad request
    # never is.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


class ModelNotAvailable(RuntimeError):
    """The server is reachable but does not have the model. Separate because the
    fix is one command, and burying that under a raw 404 wastes the user's time.
    """


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    total_tokens: int
    dimension: int


class Embedder:
    """Wraps the Ollama embedding endpoint and enforces one invariant: every
    vector has the same width as the first one seen, so a mid-run model change
    fails here rather than partway through a Chroma write."""

    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.dimension: int | None = None
        self.total_tokens = 0

    @retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call(self, texts: list[str]) -> EmbeddingResult:
        response = self._client.post(
            f"{self.host}/api/embed",
            json={
                "model": self.model,
                "input": texts,
                # Ollama silently truncates input past the model's context
                # otherwise, which would store a vector for half a clause and
                # report success. The longest row in this corpus is far inside
                # the window, so refusing costs nothing and catches a corpus
                # that outgrows it.
                "truncate": False,
            },
        )
        if response.status_code == 404:
            raise ModelNotAvailable(
                f"Ollama at {self.host} does not have {self.model!r} — run `ollama pull {self.model}`"
            )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings") or []
        return EmbeddingResult(
            vectors=vectors,
            # Reported by newer Ollama builds; absent is not an error, the count
            # is only ever logged.
            total_tokens=payload.get("prompt_eval_count") or 0,
            dimension=len(vectors[0]) if vectors else 0,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
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
