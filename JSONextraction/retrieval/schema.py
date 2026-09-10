"""Shared constants and ID derivation for the embedding-view schema.

Mirrors `pipeline/schema.py`'s role for `raw_extraction.json`: one place that
owns the version marker and the identity rule, so both stay consistent
instead of drifting between whatever module happens to write them.
"""
from __future__ import annotations

import hashlib

# 2.0.0 changed how `embedding_id` is derived, so ids written by 1.0.0 are not
# comparable to these. The version is part of the Chroma collection name for
# exactly that reason — a rebuild lands in a new collection instead of
# half-overwriting the old one.
EMBEDDING_SCHEMA_VERSION = "2.0.0"


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

    Deliberately NOT `node_id`: that is a positional sequential counter
    assigned in tree-build order (see `pipeline/schema.py`'s `NodeIdGenerator`
    and the `node_id_stable_*` regression checks) — stable run-to-run on
    identical input, but it shifts for every node after any upstream
    insertion/deletion, even nodes whose own content never changed. A vector
    store keyed on that would silently orphan and duplicate embeddings across
    pipeline versions.

    Every component below is load-bearing; each was added because measurement
    across the 6-specimen corpus (4021 nodes) showed the previous key silently
    collapsing rows on upsert:

    - `document_key` (the source `sha256`): without it, the SAME clause in two
      contracts shares an id. These are all one standard form, so this was the
      dominant failure — 48.6% of the corpus collided, with `rehabGedung` and
      `Rancangan Kontrak` alone sharing 643 of 729 ids.
    - `sub_document`: keeps identical path shapes in different sections apart
      (section "A" exists in both `general_terms` and an annex).
    - `page` + `path` + `label_normalized`: `path` is only the label chain, so
      several independent lists restarting at "1." inside one sub-document map
      to the same path. Page separates them.
    - `text`: separates nodes that share a position key but hold different
      content — and, deliberately, makes the id change when the text changes,
      so a re-extraction that fixes a typo produces a new id rather than
      leaving a stale vector silently attached to corrected text.
    - `occurrence`: last resort. A handful of nodes (89 of 4021) are identical
      in EVERY field above — same page, same parent, same label, same text,
      e.g. three list items under one parent all reading "Pekerjaan Pasangan
      dan Plesteran". Nothing content-derived can separate those, so they are
      numbered in document order. This reintroduces positional fragility only
      within a set of byte-identical nodes, which is harmless: if the ordinals
      shift, each id still resolves to the same text.

    Verified collision-free (4021/4021 unique) across all six specimens.
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
