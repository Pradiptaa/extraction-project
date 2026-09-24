"""Settings for the embedding/load stage, read once from `retrieval/.env`.
Owns the collection-naming rule that keeps one model's vectors out of another's
collection.

This branch runs entirely on a local Ollama; the Mistral API path lives on
`master`. Nothing here requires a credential, so a run can never be refused for
the want of one.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
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

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Ollama's own default is 2048. See Settings.num_ctx.
DEFAULT_NUM_CTX = 8192


# Chroma accepts only [A-Za-z0-9._-] in a collection name, and `__` is this
# module's own field separator — so neither may come from a model name.
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9-]")


def model_slug(model: str) -> str:
    """A model name that is safe inside a collection name.

    Ollama tags carry characters Chroma rejects (`bge-m3:latest`, and a
    Hugging Face id would add `/`). Substitution alone is not enough: `bge:m3`
    and `bge/m3` would collapse onto one collection and silently mix two
    models' vectors. So a name that had to be changed carries a digest of the
    original, which keeps the mapping one-to-one.

    A name that is already safe is returned untouched, so collections written
    before this existed keep their names.
    """
    safe = _UNSAFE_IN_NAME.sub("-", model)
    if safe == model:
        return safe
    return f"{safe}-{hashlib.sha1(model.encode('utf-8')).hexdigest()[:6]}"


def collection_name(
    prefix: str,
    model: str,
    schema_version: str = EMBEDDING_SCHEMA_VERSION,
    index_tag: str | None = INDEX_TAG,
) -> str:
    """`contracts__bge-m3-latest-1f0c2a__v2_1_0__hnsw-m64ef400`.

    Encoding the model and schema version routes incompatible rows to separate
    collections, which lets old and new coexist while a re-embed runs. The
    `index_tag` is not a compatibility guard but keeps two differently-indexed
    collections side by side; pass None for one written before tagging existed.
    """
    tag = f"__{index_tag}" if index_tag else ""
    return f"{prefix}__{model_slug(model)}__v{schema_version.replace('.', '_')}{tag}"


@dataclass(frozen=True)
class Settings:
    model: str
    batch_size: int
    db_path: Path
    collection: str
    # Empty unless synthesis is configured; retrieval never reads it.
    chat_model: str = ""
    host: str = DEFAULT_OLLAMA_HOST
    # Ollama defaults to 2048 and silently truncates past it — which would drop
    # retrieved clauses out of the prompt without a word. Always sent explicitly.
    num_ctx: int = DEFAULT_NUM_CTX
    # Where to embed a *query*. On by default because the embedding and chat
    # models do not fit in 6 GB together, and sharing the GPU makes them evict
    # each other on every question. Turn it off on a card that holds both.
    # Bulk loading ignores this and uses the GPU, which is 2.8x faster there.
    query_embed_on_cpu: bool = True

    @property
    def query_num_gpu(self) -> int | None:
        """What `Embedder` should ask for when embedding a question."""
        return 0 if self.query_embed_on_cpu else None

    @property
    def model_slug(self) -> str:
        """The model as it appears inside `collection`."""
        return model_slug(self.model)


# Relative CHROMA_DB_PATH values resolve from here, not the cwd, which would
# create a second empty store outside the gitignored one.
PROJECT_DIR = ENV_PATH.parent.parent


def resolve_db_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def load_settings() -> Settings:
    """No argument and no credential check: every model this branch uses is
    served locally, so there is nothing a run can lack permission to reach."""
    load_dotenv(ENV_PATH)

    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model:
        raise SystemExit("EMBEDDING_MODEL is not set — it must be pinned, never defaulted silently.")

    prefix = os.getenv("CHROMA_COLLECTION_PREFIX", "contracts").strip()
    db_path = resolve_db_path(os.getenv("CHROMA_DB_PATH", "./chroma_data"))

    return Settings(
        model=model,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        db_path=db_path,
        collection=collection_name(prefix, model),
        chat_model=os.getenv("CHAT_MODEL", "").strip(),
        host=os.getenv("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_HOST,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX") or DEFAULT_NUM_CTX),
        query_embed_on_cpu=(os.getenv("QUERY_EMBED_ON_CPU", "true").strip().lower()
                            not in ("0", "false", "no")),
    )
