"""Opening the configured Chroma collection for reading. Its own module so
`ask` and `retrieval_evaluate` share it without importing each other."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from . import registry
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


def scan_corpus_documents(collection, views_dir: Path | None = None) -> dict[str, str]:
    """Every `document_key` present in the collection, with its name, read by
    scanning every row's metadata.

    Correct whatever else is or is not on disk, and linear in the size of the
    corpus — 349 ms over 4,686 rows, and a question asks for it. `registry.py`
    exists to answer the same question from an indexed table; this stays as the
    answer of last resort.
    """
    got = collection.get(include=["metadatas"])
    names = document_names(views_dir)
    keys = {str(m.get("document_key") or "") for m in (got.get("metadatas") or []) if m}
    return {key: names.get(key, "") for key in sorted(keys) if key}


def _registry_is_current(settings: Settings, collection) -> bool:
    """Whether the registry may be trusted for this collection right now.

    Trust is not optional. A registry that is merely *incomplete* would hand
    back fewer documents than the collection holds, and a question would be
    scoped to a subset and answered confidently from it — worse than having no
    registry at all. So the summed row counts are checked against the
    collection before a single name is used.

    An absent registry is normal and says nothing; one that exists but does
    not match is worth a warning, since someone can fix it.
    """
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
    """Every `document_key` present in the collection, with its name.

    With `settings`, the indexed catalogue is tried first and the scan is the
    fallback; without it, the scan is all there is. The two return the same
    shape and, while the registry is current, the same contents — which is what
    lets every caller keep working whether or not one has been built.
    """
    if settings is not None and _registry_is_current(settings, collection):
        registered = registry.read(
            settings.db_path, lambda c: registry.names(c, settings.collection), default=None
        )
        if registered is not None:
            return registered
    return scan_corpus_documents(collection, views_dir)


def filename_resolver(available: dict[str, str]):
    """Resolution with no registry: containment over filenames, which is all a
    scanned catalogue knows. Filenames only whatever the caller asks, so
    `filenames_only` is accepted and has nothing further to restrict."""

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
    """Resolution against the indexed catalogue.

    A fresh connection per call rather than one held open for the resolver's
    lifetime: SQLite opens in well under a millisecond, and a long-lived handle
    would have to be closed by whoever happened to stop using it.

    If a read fails after the registry was judged current — corruption, a
    table dropped by hand, a lock — the answer comes from `fallback()` instead.
    Returning no matches would be worse than wrong: a strong cue would then
    report that the file does not exist.
    """

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
    """How to turn a typed name into the documents it could mean.

    Registry-backed when one is current, which brings contract numbers, party
    names and contract titles into what can be matched; otherwise the filename
    containment the scan supports. Both answer with a few candidates and a
    total, never a catalogue — the point being that at scale the catalogue
    cannot be printed.

    The choice is made once, on the first call, because this module already
    owns it — and *lazily*, because most questions name no document. Deciding
    eagerly would mean the fallback path scanning the whole corpus to build a
    catalogue that is then never consulted, which is the cost this is all
    supposed to remove.
    """
    chosen: list = []
    scanned: list = []

    def scan_resolver():
        # Built at most once, and only if something actually needs it.
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
    """What is available, truncated.

    Printing every document was the only way to explain a miss while the
    catalogue was six long. It is not an explanation at a thousand.
    """
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
    """Turn `--document` text into the `document_key`s to search.

    Accepts a filename fragment, a `document_key` prefix, or a comma-separated
    list of either. A filename is matched the way a name in a question is —
    letters and digits only — so `--document "pembangunan rumah"` finds
    `pembangunanRumah.pdf` exactly as "pada file pembangunan rumah" does. The
    extension may be typed or not.

    Ambiguity is refused rather than guessed, since scoping to the wrong
    contract is what this feature prevents. A miss is not answered with the
    catalogue: at scale it is too long to be an answer.
    """
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
        # Against the whole filename, extension included, so a term that
        # carries `.pdf` still matches; every earlier raw-substring match is
        # still a match under this rule.
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
    """One-line human description of a resolved scope, for command output."""
    return ", ".join(name or key[:12] for key, name in scope.items())
