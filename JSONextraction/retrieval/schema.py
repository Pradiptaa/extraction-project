"""Shared constants and ID derivation for the embedding-view schema."""
from __future__ import annotations

import hashlib

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
