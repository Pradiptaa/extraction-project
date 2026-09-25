"""An indexed catalogue of the documents in a collection.

Answering "which contract is this question about" currently means reading every
row's metadata out of Chroma (`store.corpus_documents`), parsing every embedding
view on disk (`store.document_names`), and parsing every raw extraction
(`lookup.load_raw_documents`). Measured on six specimens that is ~0.4 s; the
cost is per question and linear in the corpus, so at a thousand documents it is
over a minute before retrieval starts, and the only way to explain an ambiguous
name is to print the whole catalogue.

This is the same information as a table that can be queried instead of scanned.
Nothing here extracts anything: every field is a projection of what the
pipeline already wrote into `<pdf-stem>_raw.json`.

**Derived, never authoritative.** Chroma decides what is loaded — the same rule
`load.py` applies to its manifest. A registry that is missing, stale or
corrupt must only ever cost speed, so callers fall back to scanning and
`rebuild` reconstructs it from the raw files.

    python -m retrieval.registry rebuild --collection contracts_rel__bge-m3__v2_1_0__hnsw-m64ef400
    python -m retrieval.registry list
    python -m retrieval.registry search "pembangunan rumah"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import PROJECT_DIR, load_settings

logger = logging.getLogger("retrieval.registry")

REGISTRY_SCHEMA_VERSION = "1.0.0"

# Beside the store it describes: one registry per Chroma directory, covering
# every collection in it. `chroma_data/` is gitignored at any depth, so a
# registry cannot be committed by accident.
REGISTRY_FILENAME = "registry.sqlite3"

RAW_DIR = PROJECT_DIR / "output" / "raw"

_NOT_ALNUM = re.compile(r"[^a-z0-9]")


def normalize_name(text: str) -> str:
    """Letters and digits only, lowercased.

    Filenames here are camelCase (`pembangunanRumah.pdf`) while people type
    words ("pembangunan rumah"), and one carries a space the other does not.
    Dropping everything else makes those the same string.

    The canonical copy of a rule `documents.normalize` also applies; a test
    pins the two together until that module reads this table instead.
    """
    return _NOT_ALNUM.sub("", text.lower())


def filename_stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def portable_path(path: Path) -> str:
    """A raw file's path as stored: relative to the project when it lies inside
    it, so moving the project — or syncing it to a machine where it lives
    somewhere else — does not invalidate every stored path at once."""
    try:
        return Path(path).resolve().relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def absolute_path(stored: str) -> Path:
    """The inverse of `portable_path`. An absolute path written by an earlier
    version is honoured as it is."""
    path = Path(stored)
    return path if path.is_absolute() else PROJECT_DIR / path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per (document, collection): a re-embed under a new model gives the
-- same document a second row, so old and new coexist exactly as the two
-- collections do.
CREATE TABLE IF NOT EXISTS documents (
    document_key    TEXT NOT NULL,
    collection      TEXT NOT NULL,
    filename        TEXT NOT NULL,
    -- Letters and digits only. FTS5 tokenises on word boundaries and cannot
    -- see inside `pembangunanRumah`, so a name typed as "pembangunan rumah"
    -- finds that file only when some other field happens to spell it out.
    -- `kontrakJasa.pdf` has no contract name at all, and "jasa" misses it
    -- entirely. This column keeps the containment match that does work.
    filename_norm   TEXT NOT NULL DEFAULT '',
    contract_number TEXT,
    contract_name   TEXT,
    document_type   TEXT,
    year            TEXT,
    page_count      INTEGER,
    raw_path        TEXT,
    row_count       INTEGER,
    status          TEXT,
    confidence      REAL,
    loaded_at       TEXT,
    PRIMARY KEY (document_key, collection)
);

CREATE INDEX IF NOT EXISTS documents_by_collection ON documents (collection);
CREATE INDEX IF NOT EXISTS documents_by_norm ON documents (filename_norm);

-- Belongs to the document, not to a collection: the parties do not change
-- because the text was embedded by a different model.
CREATE TABLE IF NOT EXISTS document_parties (
    document_key TEXT NOT NULL,
    role         TEXT,
    organization TEXT,
    person       TEXT
);

CREATE INDEX IF NOT EXISTS parties_by_document ON document_parties (document_key);

-- What turns "which contract did you mean" into a ranked query with a count,
-- instead of a scan that can only explain itself by listing everything.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    document_key UNINDEXED,
    filename,
    contract_number,
    contract_name,
    organizations,
    tokenize='unicode61'
);
"""


@dataclass
class Party:
    role: str = ""
    organization: str = ""
    person: str = ""


@dataclass
class Document:
    """One row of `documents`, plus the parties used to search for it.

    Every field but `document_key` and `filename` may be empty: a blank
    template is a faithful extraction of a blank template, not a failure, and
    several specimens carry no parties and no dates at all.
    """

    document_key: str
    filename: str
    contract_number: str = ""
    contract_name: str = ""
    document_type: str = ""
    year: str = ""
    page_count: int | None = None
    raw_path: str = ""
    row_count: int | None = None
    status: str = ""
    confidence: float | None = None
    # When a load last put this document in the collection. Set by `load.py`,
    # empty for a row built by a command-line rebuild.
    loaded_at: str = ""
    parties: list[Party] = field(default_factory=list)

    @property
    def organizations(self) -> str:
        """The party text the FTS index searches, as one string."""
        return " ".join(
            part for party in self.parties for part in (party.organization, party.person) if part
        )

    def describe(self) -> str:
        """One line that distinguishes this document from a near-namesake.

        The point of the registry at scale: five lines of `Rancangan
        Kontrak.pdf` are useless, five lines carrying the contract number, the
        organisation and the year are not.
        """
        bits = [self.filename]
        if self.contract_number:
            bits.append(self.contract_number)
        organization = next((p.organization for p in self.parties if p.organization), "")
        # The specimens mark a template footnote with a leading `*)`; it is
        # faithful to the source and noise in a one-line description.
        organization = organization.removeprefix("*)").strip()
        if organization:
            bits.append(organization if len(organization) <= 48 else organization[:45] + "...")
        if self.year:
            bits.append(self.year)
        return " — ".join(bits)


# --------------------------------------------------------------------------
# projection from the raw extraction
# --------------------------------------------------------------------------

def _value(core: dict, field_name: str):
    """`core.<field>.value`, tolerating a field the profile did not populate."""
    entry = core.get(field_name)
    return entry.get("value") if isinstance(entry, dict) else None


def _text(value) -> str:
    return " ".join(str(value).split()) if value else ""


def _year(core: dict) -> str:
    """The budget year, when the extraction states one.

    Only `precision: "year"` entries are considered. The date list also holds
    day-precision values recovered from things like a `1.03.10` account code,
    so taking the earliest date would routinely print 2010 for a 2023 contract.
    An unknown year is left empty rather than guessed at.
    """
    for entry in _value(core, "key_dates") or []:
        if isinstance(entry, dict) and entry.get("precision") == "year":
            year = _text(entry.get("date"))
            if year:
                return year
    return ""


def _parties(core: dict) -> list[Party]:
    parties: list[Party] = []
    for entry in _value(core, "parties") or []:
        if not isinstance(entry, dict):
            continue
        organization = (entry.get("organization") or {}).get("value")
        representative = entry.get("representative") or {}
        party = Party(
            role=_text(entry.get("role") or entry.get("role_label")),
            organization=_text(organization),
            person=_text(representative.get("name")),
        )
        # A party with nothing filled in is a template blank; it cannot help
        # anyone tell two documents apart, so it is not stored.
        if party.organization or party.person:
            parties.append(party)
    return parties


def project(raw: dict, raw_path: Path | None = None) -> Document | None:
    """One parsed `*_raw.json` as a registry row, or None without a key.

    `source.sha256` is the `document_key` Chroma stores, which is what makes
    this table joinable to the collection at all.
    """
    source = raw.get("source") or {}
    document_key = source.get("sha256")
    if not document_key:
        return None
    core = raw.get("core") or {}
    status = core.get("_status") or {}
    confidence = status.get("overall_confidence")
    return Document(
        document_key=document_key,
        filename=_text(source.get("file")) or document_key[:12],
        contract_number=_text(_value(core, "contract_number")),
        contract_name=_text(_value(core, "contract_name")),
        document_type=_text(_value(core, "document_type")),
        year=_year(core),
        page_count=source.get("page_count") if isinstance(source.get("page_count"), int) else None,
        raw_path=portable_path(raw_path) if raw_path else "",
        status=_text(status.get("document_status")),
        confidence=confidence if isinstance(confidence, (int, float)) else None,
        parties=_parties(core),
    )


def read_raw_documents(raw_dir: Path | None = None):
    """Yield `(path, Document)` for every raw extraction on disk.

    The only full pass over the raw files, and it happens at build time rather
    than once per question — which is the whole point of the table.
    """
    directory = RAW_DIR if raw_dir is None else raw_dir
    if not directory.is_dir():
        logger.warning("no raw extraction directory at %s", directory)
        return
    # Only `*_raw.json`: the directory also holds older files under pre-fix names.
    for path in sorted(directory.glob("*_raw.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read %s for the registry (%s)", path.name, exc)
            continue
        document = project(raw, path)
        if document is None:
            logger.warning("%s has no source.sha256 — cannot be registered", path.name)
            continue
        yield path, document


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def registry_path(db_path: Path) -> Path:
    return db_path / REGISTRY_FILENAME


# Every column `_SCHEMA` defines on `documents`. Compared against what is on
# disk, because `CREATE TABLE IF NOT EXISTS` leaves an older table untouched
# and the first query against it then fails.
_DOCUMENT_COLUMNS = {
    "document_key", "collection", "filename", "filename_norm", "contract_number",
    "contract_name", "document_type", "year", "page_count", "raw_path", "row_count",
    "status", "confidence", "loaded_at",
}


def _is_stale(connection: sqlite3.Connection) -> bool:
    """Whether what is on disk is an older shape than this module writes.

    The version is checked first and the columns second, because a version is
    only as reliable as the memory of whoever changed the schema.
    """
    tables = {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    if not tables:
        return False  # a new file, not a stale one
    if "meta" not in tables or "documents" not in tables:
        return True
    stored = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if not stored or stored["value"] != REGISTRY_SCHEMA_VERSION:
        return True
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(documents)")}
    return not _DOCUMENT_COLUMNS.issubset(columns)


def connect(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) a registry, with its schema applied.

    A registry written by an older version of this module is dropped and
    recreated empty rather than migrated. That is only acceptable because the
    table is derived: everything in it can be rebuilt from the raw files, so
    the cost of being wrong is a rebuild, never data. Migrating instead would
    mean carrying a migration per schema change for a cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    if _is_stale(connection):
        logger.warning(
            "%s was written by an older registry schema — resetting it. "
            "Run `python -m retrieval.registry rebuild` to refill it.", path.name,
        )
        connection.executescript(
            """DROP TABLE IF EXISTS documents;
               DROP TABLE IF EXISTS document_parties;
               DROP TABLE IF EXISTS documents_fts;
               DROP TABLE IF EXISTS meta;"""
        )
        connection.commit()
    connection.executescript(_SCHEMA)
    connection.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (REGISTRY_SCHEMA_VERSION,),
    )
    connection.commit()
    return connection


def try_open(path: Path) -> sqlite3.Connection | None:
    """An existing, current registry, or None.

    The reader's counterpart to `connect`. A reader must not create a registry
    that was never built, and must not reset one written by an older schema —
    `connect` does both, which is right for a command that is about to fill it
    and wrong for a question that merely wants to look something up. Anything
    unusable reads as None, and the caller falls back to scanning.
    """
    if not path.exists():
        return None
    connection = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        # `_is_stale` calls a file with no tables "new, not stale", which is
        # right for `connect` — it is about to create them — and wrong here,
        # where nothing can be created. A zero-byte file is what an interrupted
        # `connect` leaves behind, and reading it would fail on the first query.
        has_tables = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()
        if not has_tables or _is_stale(connection):
            logger.warning(
                "%s is empty or was written by an older registry schema — scanning instead. "
                "Run `python -m retrieval.registry rebuild` to refresh it.", path.name,
            )
            connection.close()
            return None
        return connection
    except sqlite3.Error as exc:
        if connection is not None:
            connection.close()
        logger.warning("could not read %s (%s) — scanning instead", path.name, exc)
        return None


def read(db_path: Path, reader, default=None):
    """`reader(connection)` against the registry, or `default` if it cannot run.

    The one place a reader's failure policy lives. Opening is not the only thing
    that can fail — a corrupt page, a table dropped by hand, a lock held by a
    concurrent load all surface on a later query — and every one of them must
    cost a scan, never a traceback, because this table is a cache of what the
    scan would say. `default` is how the caller spells "fall back".
    """
    connection = try_open(registry_path(db_path))
    if connection is None:
        return default
    try:
        return reader(connection)
    except sqlite3.Error as exc:
        logger.warning("could not read the document registry (%s) — scanning instead", exc)
        return default
    finally:
        connection.close()


def schema_version(connection: sqlite3.Connection) -> str:
    row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return row["value"] if row else ""


def upsert(connection: sqlite3.Connection, document: Document, collection: str,
           loaded_at: str = "") -> None:
    """Record one document as present in one collection.

    Idempotent, because `load.py` is resumable and will re-register a document
    whose batches are retried.
    """
    connection.execute(
        """INSERT INTO documents (document_key, collection, filename, filename_norm,
                                  contract_number, contract_name, document_type, year,
                                  page_count, raw_path, row_count, status, confidence, loaded_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(document_key, collection) DO UPDATE SET
               filename=excluded.filename, filename_norm=excluded.filename_norm,
               contract_number=excluded.contract_number,
               contract_name=excluded.contract_name, document_type=excluded.document_type,
               year=excluded.year, page_count=excluded.page_count, raw_path=excluded.raw_path,
               row_count=COALESCE(excluded.row_count, documents.row_count),
               status=excluded.status, confidence=excluded.confidence,
               loaded_at=excluded.loaded_at""",
        (document.document_key, collection, document.filename,
         normalize_name(filename_stem(document.filename)), document.contract_number,
         document.contract_name, document.document_type, document.year, document.page_count,
         document.raw_path, document.row_count, document.status, document.confidence,
         loaded_at or document.loaded_at),
    )
    # Parties and the search index belong to the document, so they are replaced
    # wholesale rather than accumulated across collections.
    connection.execute("DELETE FROM document_parties WHERE document_key = ?", (document.document_key,))
    connection.executemany(
        "INSERT INTO document_parties (document_key, role, organization, person) VALUES (?,?,?,?)",
        [(document.document_key, p.role, p.organization, p.person) for p in document.parties],
    )
    connection.execute("DELETE FROM documents_fts WHERE document_key = ?", (document.document_key,))
    connection.execute(
        """INSERT INTO documents_fts (document_key, filename, contract_number,
                                      contract_name, organizations)
           VALUES (?,?,?,?,?)""",
        (document.document_key, document.filename, document.contract_number,
         document.contract_name, document.organizations),
    )
    connection.commit()


def rebuild(connection: sqlite3.Connection, collection: str, raw_dir: Path | None = None,
            only_keys: set[str] | None = None, row_counts: dict[str, int] | None = None,
            loaded_at: str = "", prune: bool = False) -> int:
    """Register raw extractions on disk against `collection`.

    `only_keys` restricts it to the documents a collection actually holds;
    without it every raw file on disk is registered, which is right for a
    catalogue built before anything was loaded and wrong for one meant to
    mirror a collection. The caller decides, because only it can afford to ask
    Chroma what is in there.

    `row_counts` carries how many rows each document has in the collection.

    `prune` makes it a rebuild rather than an update: every row this
    collection holds for a document outside `only_keys` is deleted. Without it
    a document that left the collection stays registered for ever, its rows
    are counted against a collection that no longer has them, and the registry
    can never be current again — not even after a rebuild. `load.py` must not
    prune, because a load of one view knows about one document, not all of
    them.
    """
    if prune and only_keys is None:
        raise ValueError("prune needs only_keys: without them there is nothing to keep")
    registered = 0
    for _, document in read_raw_documents(raw_dir):
        if only_keys is not None and document.document_key not in only_keys:
            continue
        if row_counts is not None:
            document.row_count = row_counts.get(document.document_key)
        upsert(connection, document, collection, loaded_at)
        registered += 1
    if prune:
        removed = _prune(connection, collection, only_keys)
        if removed:
            logger.info("removed %d documents no longer in %s", removed, collection)
    logger.info("registered %d documents in %s", registered, collection)
    return registered


def _prune(connection: sqlite3.Connection, collection: str, keep: set[str]) -> int:
    """Delete this collection's rows for every document not in `keep`.

    Parties and the search index are keyed by document, not by collection, so
    they go only once no collection still registers the document — another
    collection may hold the same contract under a different model.
    """
    stale = [
        row["document_key"]
        for row in connection.execute(
            "SELECT document_key FROM documents WHERE collection = ?", (collection,)
        )
        if row["document_key"] not in keep
    ]
    for key in stale:
        connection.execute(
            "DELETE FROM documents WHERE document_key = ? AND collection = ?", (key, collection)
        )
        still_used = connection.execute(
            "SELECT 1 FROM documents WHERE document_key = ? LIMIT 1", (key,)
        ).fetchone()
        if not still_used:
            connection.execute("DELETE FROM document_parties WHERE document_key = ?", (key,))
            connection.execute("DELETE FROM documents_fts WHERE document_key = ?", (key,))
    connection.commit()
    return len(stale)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _to_document(row: sqlite3.Row, parties: list[Party] | None = None) -> Document:
    return Document(
        document_key=row["document_key"], filename=row["filename"],
        contract_number=row["contract_number"] or "", contract_name=row["contract_name"] or "",
        document_type=row["document_type"] or "", year=row["year"] or "",
        page_count=row["page_count"], raw_path=row["raw_path"] or "",
        row_count=row["row_count"], status=row["status"] or "",
        confidence=row["confidence"], loaded_at=row["loaded_at"] or "",
        parties=parties or [],
    )


def parties_of(connection: sqlite3.Connection, document_key: str) -> list[Party]:
    return [
        Party(role=r["role"] or "", organization=r["organization"] or "", person=r["person"] or "")
        for r in connection.execute(
            "SELECT role, organization, person FROM document_parties WHERE document_key = ?",
            (document_key,),
        )
    ]


def row_counts_from_collection(collection) -> dict[str, int]:
    """Rows per document, read out of the collection itself.

    This is the expensive question the registry exists to answer cheaply — one
    pass over every row's metadata — so it belongs to maintenance commands
    only, never to a question. `load.py` gets the same numbers for free as it
    loads, which is why it does not call this.

    Takes the collection rather than importing `store`, because `store` will
    import this module and the cycle would be real.
    """
    got = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in got.get("metadatas") or []:
        key = (metadata or {}).get("document_key")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def is_current(connection: sqlite3.Connection, collection: str, collection_rows: int) -> bool:
    """Whether the registry accounts for exactly the rows the collection holds.

    The guard that makes reading the registry instead of scanning safe. A
    registry that is merely *incomplete* is the dangerous case: it would answer
    with fewer documents than exist, and a question would be scoped to a subset
    and answered confidently from it. A missing registry announces itself; a
    half-full one does not.

    Both sides are cheap: a summed column against `collection.count()`. A
    document whose `row_count` was never recorded makes the total unverifiable,
    which counts as not current — the caller falls back to scanning rather than
    trusting a number it cannot check.
    """
    row = connection.execute(
        """SELECT COUNT(*) AS documents,
                  COUNT(row_count) AS counted,
                  COALESCE(SUM(row_count), 0) AS rows_total
             FROM documents WHERE collection = ?""",
        (collection,),
    ).fetchone()
    if not row or not row["documents"]:
        return False
    if row["counted"] != row["documents"]:
        logger.debug("registry has %d documents but only %d carry a row count",
                     row["documents"], row["counted"])
        return False
    return row["rows_total"] == collection_rows


def count(connection: sqlite3.Connection, collection: str) -> int:
    row = connection.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE collection = ?", (collection,)
    ).fetchone()
    return row["n"] if row else 0


def names(connection: sqlite3.Connection, collection: str) -> dict[str, str]:
    """`document_key` -> filename, the shape `store.corpus_documents` returns."""
    return {
        row["document_key"]: row["filename"]
        for row in connection.execute(
            "SELECT document_key, filename FROM documents WHERE collection = ? ORDER BY filename",
            (collection,),
        )
    }


def raw_paths(connection: sqlite3.Connection, collection: str) -> dict[str, Path]:
    """`document_key` -> absolute path of its raw extraction, for every
    document that has one. One query, so a reader fetching several documents
    does not pay a lookup per document."""
    return {
        row["document_key"]: absolute_path(row["raw_path"])
        for row in connection.execute(
            "SELECT document_key, raw_path FROM documents WHERE collection = ? AND raw_path != ''",
            (collection,),
        )
    }


def get(connection: sqlite3.Connection, document_key: str, collection: str) -> Document | None:
    row = connection.execute(
        "SELECT * FROM documents WHERE document_key = ? AND collection = ?",
        (document_key, collection),
    ).fetchone()
    return _to_document(row, parties_of(connection, document_key)) if row else None


@dataclass
class Matches:
    """What a name resolved to, and how much was not shown.

    `total` is deliberately separate from `documents`: at scale the useful
    reply to an ambiguous name is a handful of candidates and a count, never
    the catalogue.
    """

    documents: list[Document] = field(default_factory=list)
    total: int = 0

    @property
    def truncated(self) -> int:
        return max(0, self.total - len(self.documents))


def find_by_filename(connection: sqlite3.Connection, text: str, collection: str) -> Matches:
    """Documents whose filename contains `text`, both normalised.

    The complement to `search`: FTS5 cannot see inside a camelCase filename,
    and this cannot rank or match a contract number. Together they cover what
    people actually type.

    A scan, but over the documents table rather than over every row's metadata
    in Chroma — thousands of short strings instead of millions of rows, which
    is the difference the registry exists to make. `total` counts every match;
    the caller decides how many to show.
    """
    needle = normalize_name(text)
    if not needle:
        return Matches()
    rows = connection.execute(
        """SELECT * FROM documents
            WHERE collection = ? AND filename_norm LIKE '%' || ? || '%'
         ORDER BY LENGTH(filename_norm), filename""",
        (collection, needle),
    ).fetchall()
    return Matches(
        documents=[_to_document(r, parties_of(connection, r["document_key"])) for r in rows],
        total=len(rows),
    )


def resolve(connection: sqlite3.Connection, text: str, collection: str,
            limit: int = 5, filenames_only: bool = False) -> Matches:
    """What a typed name refers to: the filename match if there is one, else
    the ranked metadata search.

    Filename first because it is what `--document` has always matched and what
    people name a file by; the FTS search then covers contract numbers, party
    names and contract titles, which is how documents are told apart once
    filenames stop being unique.

    `filenames_only` skips the metadata search. It exists for text that was
    not clearly offered as a name: organisation names are made of ordinary
    words — "Satuan Kerja Dinas Tenaga Kerja" — so searching them with a
    fragment of an ordinary sentence finds a contract the sentence never named.
    """
    by_name = find_by_filename(connection, text, collection)
    if by_name.total or filenames_only:
        return Matches(documents=by_name.documents[:limit], total=by_name.total)
    return search(connection, text, collection, limit)


def _fts_query(text: str) -> str:
    """User text as an FTS5 prefix query.

    Every token is quoted, because a contract number (`08/PUPRPRKP-B.PNK`) is
    full of characters FTS5 would otherwise read as operators.
    """
    tokens = [t for t in "".join(c if c.isalnum() else " " for c in text).split() if t]
    return " ".join(f'"{token}"*' for token in tokens)


def search(connection: sqlite3.Connection, text: str, collection: str,
           limit: int = 5) -> Matches:
    """Documents matching `text`, best first, with the total match count.

    Ranked by FTS5's own relevance, so a full filename beats a shared word.
    """
    query = _fts_query(text)
    if not query:
        return Matches()
    rows = connection.execute(
        """SELECT d.*, f.rank AS rank
             FROM documents_fts f
             JOIN documents d ON d.document_key = f.document_key
            WHERE documents_fts MATCH ? AND d.collection = ?
         ORDER BY f.rank
            LIMIT ?""",
        (query, collection, limit),
    ).fetchall()
    total = connection.execute(
        """SELECT COUNT(*) AS n
             FROM documents_fts f
             JOIN documents d ON d.document_key = f.document_key
            WHERE documents_fts MATCH ? AND d.collection = ?""",
        (query, collection),
    ).fetchone()["n"]
    return Matches(
        documents=[_to_document(r, parties_of(connection, r["document_key"])) for r in rows],
        total=total,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _collection_counts(settings, collection: str) -> tuple[dict[str, int] | None, int]:
    """Rows per document for `collection`, or None when it cannot be opened.

    A registry is routinely built before anything is loaded, so a missing
    collection is a state to report, not an error to raise.
    """
    # Deferred: `store` imports this module once the catalogue is read from it.
    import dataclasses

    from .store import open_collection

    try:
        opened = open_collection(dataclasses.replace(settings, collection=collection))
    except SystemExit as exc:
        logger.warning("cannot read %s to count rows (%s)", collection, exc)
        return None, 0
    counts = row_counts_from_collection(opened)
    return counts, sum(counts.values())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="The document catalogue for a Chroma store")
    parser.add_argument("action", choices=("rebuild", "list", "search", "status"))
    parser.add_argument("text", nargs="?", help="What to search for")
    parser.add_argument("--collection", help="Default: the configured collection")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)

    settings = load_settings()
    collection = args.collection or settings.collection
    connection = connect(registry_path(settings.db_path))

    if args.action == "rebuild":
        # Counted from the collection here, because a rebuild is maintenance
        # and can afford the full read that a question cannot. Without the
        # counts the registry cannot be checked against the collection, and
        # anything reading it would fall back to scanning for ever.
        counts, rows = _collection_counts(settings, collection)
        if counts is None:
            registered = rebuild(connection, collection)
        else:
            # Only what the collection holds, and nothing else: a raw file
            # extracted but never loaded would otherwise be registered with no
            # row count and keep the registry unverifiable, and a document
            # since removed would be counted against rows that are gone.
            registered = rebuild(connection, collection, only_keys=set(counts),
                                 row_counts=counts, prune=True)
        print(f"{registered} documents registered in {collection}")
        if counts is None:
            print("row counts unavailable — the registry cannot be verified against the "
                  "collection, and readers will keep scanning. Load the collection first.")
        else:
            print(f"{rows} rows accounted for; "
                  f"registry is {'current' if is_current(connection, collection, rows) else 'NOT current'}")
        return 0

    if args.action == "status":
        counts, rows = _collection_counts(settings, collection)
        documents = count(connection, collection)
        print(f"collection : {collection}")
        print(f"registry   : {documents} document(s)")
        if counts is None:
            print("collection : unavailable — cannot verify")
            return 1
        print(f"collection : {len(counts)} document(s), {rows} rows")
        current = is_current(connection, collection, rows)
        print(f"verdict    : {'current' if current else 'stale — readers will scan instead'}")
        if not current:
            print("            run `python -m retrieval.registry rebuild`")
        return 0 if current else 1

    if args.action == "list":
        documents = names(connection, collection)
        print(f"{len(documents)} documents in {collection}")
        for key in documents:
            entry = get(connection, key, collection)
            if entry:
                print(f"  {key[:12]}  {entry.describe()}")
        return 0

    if not args.text:
        parser.error("search needs something to search for")
    matches = search(connection, args.text, collection, args.limit)
    if not matches.total:
        print(f"no document matches {args.text!r}")
        return 1
    print(f"{matches.total} match(es) for {args.text!r}:")
    for document in matches.documents:
        print(f"  {document.document_key[:12]}  {document.describe()}")
    if matches.truncated:
        print(f"  ...and {matches.truncated} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
