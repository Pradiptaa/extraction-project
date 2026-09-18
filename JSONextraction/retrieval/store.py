"""Opening the configured Chroma collection for reading. Its own module so
`ask` and `retrieval_evaluate` share it without importing each other."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from .config import PROJECT_DIR, Settings

logger = logging.getLogger(__name__)

# Where `build_embedding_view --out` writes. Only used to put readable names on
# `document_key` values; scoping works without it.
VIEWS_DIR = PROJECT_DIR / "output" / "embedding"


def open_collection(settings: Settings):
    """The configured collection, or SystemExit with the fix spelled out — so a
    missing or empty collection doesn't read as a catastrophic regression."""
    if not settings.db_path.exists():
        raise SystemExit(
            f"no Chroma store at {settings.db_path} — run `python -m retrieval.load` first"
        )
    client = chromadb.PersistentClient(path=str(settings.db_path))
    try:
        collection = client.get_collection(settings.collection)
    except Exception:
        available = [c.name for c in client.list_collections()]
        raise SystemExit(
            f"collection {settings.collection!r} does not exist. Present: {available or 'none'}. "
            "Run `python -m retrieval.load` for this model and schema version."
        )
    if collection.count() == 0:
        raise SystemExit(
            f"collection {settings.collection!r} is empty — nothing has been loaded into it"
        )
    logger.info("opened %s (%d rows)", settings.collection, collection.count())
    return collection


def document_names(views_dir: Path | None = None) -> dict[str, str]:
    """`document_key` -> source PDF filename, read from the embedding views.

    A convenience, not a dependency: the collection stores only the sha256, and
    everything here degrades to bare keys when the views are absent.
    """
    directory = VIEWS_DIR if views_dir is None else views_dir
    names: dict[str, str] = {}
    if not directory.is_dir():
        logger.debug("no embedding views at %s — document names unavailable", directory)
        return names
    for path in sorted(directory.glob("*_embedding_view.json")):
        try:
            source = json.loads(path.read_text(encoding="utf-8")).get("source") or {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read %s for document names (%s)", path.name, exc)
            continue
        key, name = source.get("raw_extraction_sha256"), source.get("file")
        if key and name:
            names[key] = name
    return names


def corpus_documents(collection, views_dir: Path | None = None) -> dict[str, str]:
    """Every `document_key` present in the collection, with its name. Read from
    the collection, so a view built but never loaded is not offered as a scope."""
    got = collection.get(include=["metadatas"])
    names = document_names(views_dir)
    keys = {str(m.get("document_key") or "") for m in (got.get("metadatas") or []) if m}
    return {key: names.get(key, "") for key in sorted(keys) if key}


def resolve_scope(spec: str, collection, views_dir: Path | None = None) -> dict[str, str]:
    """Turn `--document` text into the `document_key`s to search.

    Accepts a filename substring, a `document_key` prefix, or a comma-separated
    list of either, matched case-insensitively. Ambiguity is refused rather than
    guessed, since scoping to the wrong contract is what this feature prevents.
    """
    available = corpus_documents(collection, views_dir)
    if not available:
        raise SystemExit(
            "no document_key metadata in the collection — it predates document scoping. "
            "Rebuild the views and run `python -m retrieval.load`."
        )

    catalogue = "\n".join(
        f"  {key[:12]}  {name or '(name unknown — embedding views not on disk)'}"
        for key, name in available.items()
    )

    selected: dict[str, str] = {}
    for raw in spec.split(","):
        term = raw.strip()
        if not term:
            continue
        needle = term.lower()
        matches = {
            key: name
            for key, name in available.items()
            if key.lower().startswith(needle) or (name and needle in name.lower())
        }
        if not matches:
            raise SystemExit(f"no document matches {term!r}. Available:\n{catalogue}")
        if len(matches) > 1:
            listed = "\n".join(f"  {k[:12]}  {n}" for k, n in matches.items())
            raise SystemExit(
                f"{term!r} is ambiguous — it matches {len(matches)} documents:\n{listed}\n"
                "Use a longer substring or a document_key prefix."
            )
        selected.update(matches)
    if not selected:
        raise SystemExit(f"--document was empty. Available:\n{catalogue}")
    return selected


def describe_scope(scope: dict[str, str]) -> str:
    """One-line human description of a resolved scope, for command output."""
    return ", ".join(name or key[:12] for key, name in scope.items())
