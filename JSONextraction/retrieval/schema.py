"""Shared constants and ID derivation for the embedding-view schema."""
from __future__ import annotations

import hashlib

# Part of the Chroma collection name, so a rebuild lands in a new collection
# rather than half-overwriting the old one. 2.1.0 added table_row rows without
# changing id derivation, so 2.0.0 vectors can still be reused via --reuse-from.
EMBEDDING_SCHEMA_VERSION = "2.1.0"


def embedding_id(
    document_key: str,
    sub_document: str | None,
    page: int | None,
    path: list[str],
    label_normalized: str | None,
    text: str,
    occurrence: int = 0,
) -> str:
    """A collision-free, durable per-node key for the embedding view and Chroma.

    Content-derived rather than `node_id`, which is a positional counter that
    shifts for every node after an upstream insertion and would orphan vectors
    across pipeline versions. Every component is needed to stay unique:
    `document_key` separates the same clause in two contracts (these are all one
    standard form), `sub_document` and `page` separate identical path shapes,
    `text` separates same-position nodes and forces a new id when text changes,
    and `occurrence` numbers the handful of nodes identical in every other field.
    """
    raw = "::".join(
        [
            document_key,
            sub_document or "",
            str(page),
            "/".join(path),
            label_normalized or "",
            text,
            str(occurrence),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
