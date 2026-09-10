"""Shared constants and ID derivation for the embedding-view schema.

Mirrors `pipeline/schema.py`'s role for `raw_extraction.json`: one place that
owns the version marker and the identity rule, so both stay consistent
instead of drifting between whatever module happens to write them.
"""
from __future__ import annotations

import hashlib

EMBEDDING_SCHEMA_VERSION = "1.0.0"


def embedding_id(sub_document: str | None, path: list[str], label_normalized: str | None) -> str:
    """A durable per-node key for the embedding view and, eventually, Chroma.

    Deliberately NOT `node_id`: `node_id` is a positional sequential counter
    assigned in tree-build order (see `pipeline/schema.py`'s `NodeIdGenerator`
    and the `node_id_stable_*` regression checks) — stable run-to-run on
    identical input, but it shifts for every node after any upstream
    insertion/deletion, even nodes whose own content never changed. A vector
    store keyed on that would silently orphan and duplicate embeddings across
    pipeline versions.

    Built from `hierarchy_path` (`node["path"]`, already the full label chain
    from the tree root down to this node) plus `label_normalized`, scoped by
    `sub_document` so the same path shape in two different sub-documents
    (e.g. section "A" appearing in both `general_terms` and an annex) can't
    collide. A node with no numbering at all (headings, captions) has no
    `label_normalized` and often an empty `path`; those hash to a key formed
    from what they do have, which is not guaranteed unique for two
    empty-path, unlabeled nodes with otherwise-identical placement — an
    acceptable gap for this scaffolding stage, revisit if the chunker needs
    stronger guarantees there.
    """
    raw = f"{sub_document or ''}::{'/'.join(path)}::{label_normalized or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
