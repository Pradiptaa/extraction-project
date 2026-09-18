"""Settings for the embedding/load stage, read once from `retrieval/.env`.
Owns the collection-naming rule that keeps one model's vectors out of another's
collection."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .schema import EMBEDDING_SCHEMA_VERSION

ENV_PATH = Path(__file__).with_name(".env")

# Set explicitly because Chroma's defaults lose real recall on this corpus
# (recall@5 by distance 0.812 at defaults, 1.000 here). Raising them buys nothing.
INDEX_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 64,
    "hnsw:construction_ef": 400,
    "hnsw:search_ef": 200,
}

# Bump when INDEX_METADATA changes, so the rebuilt index gets its own collection.
INDEX_TAG = "hnsw-m64ef400"


def collection_name(
    prefix: str,
    model: str,
    schema_version: str = EMBEDDING_SCHEMA_VERSION,
    index_tag: str | None = INDEX_TAG,
) -> str:
    """`contracts__mistral-embed__v2_0_0__hnsw-m64ef400`.

    Encoding the model and schema version routes incompatible rows to separate
    collections, which lets old and new coexist while a re-embed runs. The
    `index_tag` is not a compatibility guard but keeps two differently-indexed
    collections side by side; pass None for one written before tagging existed.
    """
    tag = f"__{index_tag}" if index_tag else ""
    return f"{prefix}__{model}__v{schema_version.replace('.', '_')}{tag}"


@dataclass(frozen=True)
class Settings:
    # repr=False so formatting a Settings object can't leak the key into a log.
    api_key: str = field(repr=False)
    model: str
    batch_size: int
    request_delay: float
    db_path: Path
    collection: str
    # Empty unless synthesis is configured; retrieval never reads it.
    chat_model: str = ""

    @property
    def redacted_key(self) -> str:
        return f"{self.api_key[:3]}...({len(self.api_key)} chars)" if self.api_key else "MISSING"


# Relative CHROMA_DB_PATH values resolve from here, not the cwd, which would
# create a second empty store outside the gitignored one.
PROJECT_DIR = ENV_PATH.parent.parent

# Everything not listed here runs with no API key at all.
_API_RETRIEVERS = {"dense", "brute", "hybrid", "hybrid-brute"}
_API_SYNTHESIZERS = {"mistral"}


def needs_api_key(retriever: str, synthesizer: str = "null") -> bool:
    """Whether a run can reach the Mistral API. `bm25` with the null synthesizer
    cannot, so it must run without credentials."""
    return retriever in _API_RETRIEVERS or synthesizer in _API_SYNTHESIZERS


def resolve_db_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def load_settings(require_api_key: bool = True) -> Settings:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise SystemExit(
            f"MISTRAL_API_KEY is not set. Copy {ENV_PATH.name}.example to .env and fill it in."
        )

    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model:
        raise SystemExit("EMBEDDING_MODEL is not set — it must be pinned, never defaulted silently.")

    prefix = os.getenv("CHROMA_COLLECTION_PREFIX", "contracts").strip()
    db_path = resolve_db_path(os.getenv("CHROMA_DB_PATH", "./chroma_data"))

    return Settings(
        api_key=api_key,
        model=model,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        # Pacing costs seconds; tripping the free-tier cap costs a 429 cascade.
        request_delay=float(os.getenv("EMBEDDING_REQUEST_DELAY", "0.3")),
        db_path=db_path,
        collection=collection_name(prefix, model),
        chat_model=os.getenv("CHAT_MODEL", "").strip(),
    )
