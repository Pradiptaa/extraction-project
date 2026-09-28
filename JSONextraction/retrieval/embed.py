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

DEFAULT_TIMEOUT = 300.0

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
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

    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST,
                 timeout: float = DEFAULT_TIMEOUT, num_gpu: int | None = None) -> None:
        # `num_gpu=0` keeps this model off the GPU.

        self.model = model
        self.host = host.rstrip("/")
        self.num_gpu = num_gpu
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
        payload = {
            "model": self.model,
            "input": texts,
            "truncate": False,
        }
        if self.num_gpu is not None:
            payload["options"] = {"num_gpu": self.num_gpu}
        response = self._client.post(f"{self.host}/api/embed", json=payload)
        if response.status_code == 404:
            raise ModelNotAvailable(
                f"Ollama at {self.host} does not have {self.model!r} — run `ollama pull {self.model}`"
            )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings") or []
        return EmbeddingResult(
            vectors=vectors,
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
