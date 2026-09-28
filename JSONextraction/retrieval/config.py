from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .schema import EMBEDDING_SCHEMA_VERSION

ENV_PATH = Path(__file__).with_name(".env")

INDEX_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 64,
    "hnsw:construction_ef": 400,
    "hnsw:search_ef": 200,
}

INDEX_TAG = "hnsw-m64ef400"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_NUM_CTX = 8192


_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9-]")


def model_slug(model: str) -> str:
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
    tag = f"__{index_tag}" if index_tag else ""
    return f"{prefix}__{model_slug(model)}__v{schema_version.replace('.', '_')}{tag}"


@dataclass(frozen=True)
class Settings:
    model: str
    batch_size: int
    db_path: Path
    collection: str
    chat_model: str = ""
    host: str = DEFAULT_OLLAMA_HOST
    num_ctx: int = DEFAULT_NUM_CTX
    query_embed_on_cpu: bool = True

    @property
    def query_num_gpu(self) -> int | None:
        return 0 if self.query_embed_on_cpu else None

    @property
    def model_slug(self) -> str:
        return model_slug(self.model)


PROJECT_DIR = ENV_PATH.parent.parent


def resolve_db_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def load_settings() -> Settings:
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
