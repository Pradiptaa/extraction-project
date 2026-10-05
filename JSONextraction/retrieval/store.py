from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from . import registry
from .config import PROJECT_DIR, Settings

logger = logging.getLogger(__name__)

VIEWS_DIR = PROJECT_DIR / "output" / "embedding"


def open_collection(settings: Settings):
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
    logger.info("opened %s (%d rows, tree_engine=%s)", settings.collection, collection.count(),
                (collection.metadata or {}).get("tree_engine", "unstamped"))
    return collection


def document_names(views_dir: Path | None = None) -> dict[str, str]:
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
        key, name = source.get("raw_extraction_sha256"), source.get("display_name") or source.get("file")
        if key and name:
            names[key] = name
    return names


def scan_corpus_documents(collection, views_dir: Path | None = None) -> dict[str, str]:
    got = collection.get(include=["metadatas"])
    names = document_names(views_dir)
    keys = {str(m.get("document_key") or "") for m in (got.get("metadatas") or []) if m}
    return {key: names.get(key, "") for key in sorted(keys) if key}


def _registry_is_current(settings: Settings, collection) -> bool:
    current = registry.read(
        settings.db_path,
        lambda c: registry.is_current(c, settings.collection, collection.count()),
        default=None,
    )
    if current is False:
        logger.warning(
            "the document registry does not account for every row in %s — scanning "
            "instead. Run `python -m retrieval.registry rebuild` to refresh it.",
            settings.collection,
        )
    return bool(current)


def corpus_documents(collection, views_dir: Path | None = None,
                     settings: Settings | None = None) -> dict[str, str]:
    if settings is not None and _registry_is_current(settings, collection):
        registered = registry.read(
            settings.db_path, lambda c: registry.names(c, settings.collection), default=None
        )
        if registered is not None:
            return registered
    return scan_corpus_documents(collection, views_dir)


def filename_resolver(available: dict[str, str]):

    def resolve(text: str, limit: int = 5, filenames_only: bool = False) -> registry.Matches:
        needle = registry.normalize_name(text)
        if not needle:
            return registry.Matches()
        found = [
            registry.Document(document_key=key, filename=name)
            for key, name in available.items()
            if name and needle in registry.normalize_name(registry.filename_stem(name))
        ]
        found.sort(key=lambda d: (len(d.filename), d.filename))
        return registry.Matches(documents=found[:limit], total=len(found))

    return resolve


def _registry_resolver(db_path: Path, collection_name: str, fallback):
    """Resolution against the indexed catalogue. """

    def resolve(text: str, limit: int = 5, filenames_only: bool = False) -> registry.Matches:
        found = registry.read(
            db_path,
            lambda c: registry.resolve(c, text, collection_name, limit, filenames_only),
            default=None,
        )
        return found if found is not None else fallback()(text, limit, filenames_only)

    return resolve


def document_resolver(collection, views_dir: Path | None = None,
                      settings: Settings | None = None):
    """How to turn a typed name into the documents it could mean. """
    chosen: list = []
    scanned: list = []

    def scan_resolver():
        if not scanned:
            scanned.append(filename_resolver(scan_corpus_documents(collection, views_dir)))
        return scanned[0]

    def resolve(text: str, limit: int = 5, filenames_only: bool = False) -> registry.Matches:
        if not chosen:
            if settings is not None and _registry_is_current(settings, collection):
                chosen.append(_registry_resolver(settings.db_path, settings.collection,
                                                 scan_resolver))
            else:
                chosen.append(scan_resolver())
        return chosen[0](text, limit, filenames_only)

    return resolve


def _catalogue(available: dict[str, str], limit: int = 10) -> str:
    shown = list(available.items())[:limit]
    lines = [
        f"  {key[:12]}  {name or '(name unknown — embedding views not on disk)'}"
        for key, name in shown
    ]
    if len(available) > limit:
        lines.append(f"  ...and {len(available) - limit} more "
                     "(`--list-documents` to see them, `--document` to name one)")
    return "\n".join(lines)


def resolve_scope(spec: str, collection, views_dir: Path | None = None,
                  settings: Settings | None = None) -> dict[str, str]:

    available = corpus_documents(collection, views_dir, settings)
    if not available:
        raise SystemExit(
            "no document_key metadata in the collection — it predates document scoping. "
            "Rebuild the views and run `python -m retrieval.load`."
        )

    selected: dict[str, str] = {}
    for raw in spec.split(","):
        term = raw.strip()
        if not term:
            continue
        prefix = term.lower()
        needle = registry.normalize_name(term)
        matches = {
            key: name
            for key, name in available.items()
            if key.lower().startswith(prefix)
            or (name and needle and needle in registry.normalize_name(name))
        }
        if not matches:
            raise SystemExit(
                f"no document matches {term!r} — `--list-documents` shows what is in the "
                "collection, and `--list-documents --filter <text>` narrows it."
            )
        if len(matches) > 1:
            raise SystemExit(
                f"{term!r} is ambiguous — it matches {len(matches)} documents:\n"
                f"{_catalogue(matches, limit=5)}\n"
                "Use a longer fragment or a document_key prefix."
            )
        selected.update(matches)
    if not selected:
        raise SystemExit("--document was empty — `--list-documents` shows what is in the collection.")
    return selected


def describe_scope(scope: dict[str, str]) -> str:
    return ", ".join(name or key[:12] for key, name in scope.items())
