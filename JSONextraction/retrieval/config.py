"""Settings for the embedding/load stage, read once from `retrieval/.env`.

Exists so the collection-naming rule lives in exactly one place. That rule is
the guard against the failure this stage is most exposed to: embedding vectors
of one model landing in a collection built by another.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .schema import EMBEDDING_SCHEMA_VERSION

ENV_PATH = Path(__file__).with_name(".env")


def collection_name(prefix: str, model: str, schema_version: str = EMBEDDING_SCHEMA_VERSION) -> str:
    """`contracts__mistral-embed__v2_0_0`.

    Both the model and the embedding-view schema version are encoded, so a
    change to either routes writes to a different collection instead of mixing
    incompatible rows into an existing one. Chroma does reject a wrong-width
    vector on its own (`InvalidDimensionException`, verified), so this is not
    the only guard — but that error blocks the write entirely. Distinct names
    let the old and new collections coexist while a re-embed runs, instead of
    forcing a delete-then-rebuild with nothing queryable in between.

    The schema version matters independently of the model: 2.0.0 changed how
    `embedding_id` is derived, so ids from 1.0.0 address different rows even
    though the vectors are the same width.
    """
    return f"{prefix}__{model}__v{schema_version.replace('.', '_')}"


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    batch_size: int
    request_delay: float
    db_path: Path
    collection: str

    @property
    def redacted_key(self) -> str:
        return f"{self.api_key[:3]}...({len(self.api_key)} chars)" if self.api_key else "MISSING"


def load_settings() -> Settings:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            f"MISTRAL_API_KEY is not set. Copy {ENV_PATH.name}.example to .env and fill it in."
        )

    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model:
        raise SystemExit("EMBEDDING_MODEL is not set — it must be pinned, never defaulted silently.")

    prefix = os.getenv("CHROMA_COLLECTION_PREFIX", "contracts").strip()
    db_path = Path(os.getenv("CHROMA_DB_PATH", "./chroma_data")).resolve()

    return Settings(
        api_key=api_key,
        model=model,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        # A small gap between requests. The corpus is only ~63 requests, so
        # pacing costs seconds, while tripping a free-tier per-second cap costs
        # a cascade of 429s and backoff sleeps.
        request_delay=float(os.getenv("EMBEDDING_REQUEST_DELAY", "0.3")),
        db_path=db_path,
        collection=collection_name(prefix, model),
    )
