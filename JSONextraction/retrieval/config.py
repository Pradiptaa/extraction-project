"""Settings for the embedding/load stage, read once from `retrieval/.env`.

Exists so the collection-naming rule lives in exactly one place. That rule is
the guard against the failure this stage is most exposed to: embedding vectors
of one model landing in a collection built by another.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .schema import EMBEDDING_SCHEMA_VERSION

ENV_PATH = Path(__file__).with_name(".env")

# HNSW construction and search parameters, set explicitly because Chroma's
# defaults lose real recall on this corpus.
#
# Measured against an exact brute-force scan of all 4021 vectors, using the 16
# gate queries. Recall by DISTANCE is the honest metric — a row that is equally
# close is an equally good answer, while recall by id also punishes arbitrary
# tie-breaking between the many byte-identical rows:
#
#     config                    recall@5 by id   by distance
#     Chroma defaults                    0.688         0.812
#     M=32 efC=200 efS=100               0.925         ~
#     M=64 efC=400 efS=200               0.925         1.000
#     M=64 efC=500 efS=500               0.925         ~
#
# So at defaults the index genuinely did not visit the true nearest
# neighbours; at these values it always does, and raising them further buys
# nothing. The residual id-gap is duplicate tie-breaking, not recall loss.
#
# These affect only index structure, never vector compatibility, so they are NOT
# part of the dimension-safety argument behind the collection name. They are
# named in the collection anyway (see `collection_name`) because HNSW parameters
# cannot be changed in place — a change means building a new index.
INDEX_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 64,
    "hnsw:construction_ef": 400,
    "hnsw:search_ef": 200,
}

# Short, stable label for the parameter set above. Bump it when INDEX_METADATA
# changes so the rebuilt index lands in its own collection instead of being
# silently mixed with rows indexed under the old parameters.
INDEX_TAG = "hnsw-m64ef400"


def collection_name(
    prefix: str,
    model: str,
    schema_version: str = EMBEDDING_SCHEMA_VERSION,
    index_tag: str | None = INDEX_TAG,
) -> str:
    """`contracts__mistral-embed__v2_0_0__hnsw-m64ef400`.

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

    `index_tag` is there for a different reason than the other two. It is not a
    compatibility guard — the vectors are identical either way — but HNSW
    parameters cannot be altered on an existing index, so changing them means
    building a new one. Naming it keeps the two side by side and makes it
    obvious which index a recorded score came from. Pass None to address a
    collection written before index tagging existed.
    """
    tag = f"__{index_tag}" if index_tag else ""
    return f"{prefix}__{model}__v{schema_version.replace('.', '_')}{tag}"


@dataclass(frozen=True)
class Settings:
    # Excluded from the dataclass repr: the generated one printed the key in
    # full, so logging or formatting a Settings object — or a test failure
    # that displays it — would put the secret in a log. Use `redacted_key`.
    api_key: str = field(repr=False)
    model: str
    batch_size: int
    request_delay: float
    db_path: Path
    collection: str
    # Empty unless synthesis is configured. Retrieval never reads it, so the
    # whole retrieval path — including the gate — runs with no chat model set.
    chat_model: str = ""

    @property
    def redacted_key(self) -> str:
        return f"{self.api_key[:3]}...({len(self.api_key)} chars)" if self.api_key else "MISSING"


# Relative CHROMA_DB_PATH values resolve from here, not from the current
# directory. Resolving from the cwd meant running any command from the repo root
# silently created a second, empty store at HCMLProject/chroma_data — outside
# the gitignored JSONextraction/chroma_data, so a binary vector DB could be
# committed — and then reported "collection does not exist".
PROJECT_DIR = ENV_PATH.parent.parent

# Retrievers that embed the query (so call Mistral) and synthesizers that call
# a chat model. Everything else runs with no API key at all.
_API_RETRIEVERS = {"dense", "brute", "hybrid", "hybrid-brute"}
_API_SYNTHESIZERS = {"mistral"}


def needs_api_key(retriever: str, synthesizer: str = "null") -> bool:
    """Whether a run can reach the Mistral API. `bm25` with the null synthesizer
    cannot, so it must not demand a key — requiring one anyway meant the
    lexical arm of the gate could not run on a machine without credentials."""
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
        # A small gap between requests. The corpus is only ~63 requests, so
        # pacing costs seconds, while tripping a free-tier per-second cap costs
        # a cascade of 429s and backoff sleeps.
        request_delay=float(os.getenv("EMBEDDING_REQUEST_DELAY", "0.3")),
        db_path=db_path,
        collection=collection_name(prefix, model),
        chat_model=os.getenv("CHAT_MODEL", "").strip(),
    )
