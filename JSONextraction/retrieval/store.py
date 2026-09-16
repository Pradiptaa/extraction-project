"""Opening the configured Chroma collection for reading.

Its own module so the commands that query the corpus — the gate
(`retrieval_evaluate`) and `ask` — share one way to open it without one command
importing the other. `ask` used to import this from the gate module, which
coupled the user-facing CLI to the regression harness for no reason but
where the function happened to be written.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from .config import PROJECT_DIR, Settings

logger = logging.getLogger(__name__)

# Where `build_embedding_view --out` writes by convention. Only ever used to put
# human-readable filenames on `document_key` values; scoping works without it.
VIEWS_DIR = PROJECT_DIR / "output" / "embedding"


def open_collection(settings: Settings):
    """The configured collection, or SystemExit with the fix spelled out.

    Each failure is distinguished from a retrieval failure on purpose: a
    missing or empty collection would otherwise fail every query and read as a
    catastrophic regression rather than as "nothing was loaded".
    """
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

    The collection stores only the sha256, so names come from the views on
    disk. That is a convenience, not a dependency: `output/` is gitignored and
    regenerable, and every function here degrades to bare keys without it
    rather than failing. Naming a document in Chroma metadata instead would be
    cleaner, but it is a schema bump — a new collection name, and a baseline
    that no longer applies to it — which is too much to spend on cosmetics.
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
    """Every `document_key` actually present in the collection, with its name.

    Read from the collection rather than from disk, so what it reports is what
    can actually be searched. A view that was built but never loaded must not
    show up as an available scope.
    """
    got = collection.get(include=["metadatas"])
    names = document_names(views_dir)
    keys = {str(m.get("document_key") or "") for m in (got.get("metadatas") or []) if m}
    return {key: names.get(key, "") for key in sorted(keys) if key}


def resolve_scope(spec: str, collection, views_dir: Path | None = None) -> dict[str, str]:
    """Turn `--document` text into the `document_key`s to search.

    Accepts a filename substring (`rehabGedung`), a `document_key` prefix
    (`8489309d`), or a comma-separated list of either. Matching is
    case-insensitive.

    Ambiguity is refused rather than guessed: silently scoping to one of two
    matching contracts would produce a confident answer about the wrong
    document, which is the failure this whole feature exists to prevent. Every
    refusal lists what is available, following `open_collection`'s rule that an
    error names its own fix.
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
