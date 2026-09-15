"""Opening the configured Chroma collection for reading.

Its own module so the commands that query the corpus — the gate
(`retrieval_evaluate`) and `ask` — share one way to open it without one command
importing the other. `ask` used to import this from the gate module, which
coupled the user-facing CLI to the regression harness for no reason but
where the function happened to be written.
"""
from __future__ import annotations

import logging

import chromadb

from .config import Settings

logger = logging.getLogger(__name__)


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
