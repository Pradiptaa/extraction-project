"""The document catalogue: projecting raw extractions into it, and resolving a
name against it without enumerating everything.

Two properties carry the design. A blank template must register as a document
with blank fields, never as a failure — several specimens have no parties and
no dates at all. And an ambiguous name must come back as a few ranked
candidates plus a count, because the whole reason for the table is that the
catalogue is too large to print.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from retrieval.registry import (
    REGISTRY_SCHEMA_VERSION,
    Document,
    Party,
    connect,
    count,
    find_by_filename,
    get,
    is_current,
    names,
    normalize_name,
    project,
    read_raw_documents,
    rebuild,
    registry_path,
    row_counts_from_collection,
    resolve,
    schema_version,
    search,
    upsert,
)

COLLECTION = "contracts_rel__bge-m3__v2_1_0__hnsw-m64ef400"
OTHER_COLLECTION = "contracts_rel__mistral-embed__v2_1_0__hnsw-m64ef400"


def raw_document(sha: str, filename: str, number: str = "", name: str = "",
                 organization: str | None = "Dinas PUPR", person: str | None = "GIAJENG WULANDARI, ST",
                 year: str = "2023", page_count: int = 74) -> dict:
    """A raw extraction shaped as `pipeline/` writes it."""
    dates = [{"type": "unclassified_date", "date": year, "precision": "year", "confidence": 0.9}] if year else []
    # A day-precision date recovered from an account code: present in the real
    # files, and the reason the earliest date is not the contract's year.
    dates.append({"type": "unclassified_date", "date": "2010-03-01", "precision": "day", "confidence": 0.9})
    return {
        "source": {"file": filename, "sha256": sha, "page_count": page_count},
        "core": {
            "contract_number": {"value": number},
            "contract_name": {"value": name},
            "document_type": {"value": "kontrak_konstruksi"},
            "key_dates": {"value": dates},
            "parties": {"value": [{
                "role": "ppkom", "role_label": "PPKom",
                "organization": {"value": organization},
                "representative": {"name": person},
            }]},
            "_status": {"document_status": "draft_template", "overall_confidence": 0.892},
        },
    }


class ProjectionTests(unittest.TestCase):
    def test_the_document_key_is_the_one_chroma_stores(self) -> None:
        """Without `source.sha256` the table cannot be joined to the collection."""
        document = project(raw_document("abc123", "rehabGedung.pdf"))
        self.assertEqual(document.document_key, "abc123")

    def test_a_raw_file_without_a_key_is_refused_rather_than_invented(self) -> None:
        self.assertIsNone(project({"source": {"file": "x.pdf"}, "core": {}}))

    def test_core_fields_are_projected_not_re_extracted(self) -> None:
        document = project(raw_document("k", "a.pdf", number="08/PUPRPRKP-B.PNK/SP-PPK",
                                        name="Peningkatan Jalan Mekar"))
        self.assertEqual(document.contract_number, "08/PUPRPRKP-B.PNK/SP-PPK")
        self.assertEqual(document.contract_name, "Peningkatan Jalan Mekar")
        self.assertEqual(document.page_count, 74)
        self.assertEqual(document.status, "draft_template")

    def test_the_year_comes_from_a_year_precision_date(self) -> None:
        """The date list also holds day-precision values recovered from account
        codes, so the earliest date would print 2010 for a 2023 contract."""
        self.assertEqual(project(raw_document("k", "a.pdf", year="2023")).year, "2023")

    def test_an_unknown_year_is_left_empty_rather_than_guessed(self) -> None:
        self.assertEqual(project(raw_document("k", "a.pdf", year="")).year, "")

    def test_a_blank_template_registers_with_blank_fields(self) -> None:
        """An unfilled specimen is a faithful extraction, not a failed one."""
        document = project(raw_document("k", "kontrakJasa.pdf", number="", name="",
                                        organization=None, person=None, year=""))
        self.assertIsNotNone(document)
        self.assertEqual(document.filename, "kontrakJasa.pdf")
        self.assertEqual(document.contract_number, "")
        self.assertEqual(document.parties, [], "a party with nothing filled in cannot identify anything")

    def test_a_party_with_only_a_person_is_still_kept(self) -> None:
        document = project(raw_document("k", "a.pdf", organization=None, person="GATOT"))
        self.assertEqual([p.person for p in document.parties], ["GATOT"])


class DescribeTests(unittest.TestCase):
    """One line that tells two near-namesakes apart — the point of the table."""

    def test_the_description_carries_more_than_the_filename(self) -> None:
        document = project(raw_document("k", "Rancangan Kontrak.pdf",
                                        number="08/PUPRPRKP-B.PNK/SP-PPK"))
        described = document.describe()
        self.assertIn("Rancangan Kontrak.pdf", described)
        self.assertIn("08/PUPRPRKP-B.PNK/SP-PPK", described)
        self.assertIn("2023", described)

    def test_a_document_with_nothing_but_a_name_still_describes(self) -> None:
        document = project(raw_document("k", "polres.pdf", organization=None, person=None, year=""))
        self.assertEqual(document.describe(), "polres.pdf")


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.conn = connect(registry_path(self.tmp))
        self.addCleanup(self.conn.close)

    def _register(self, sha: str, filename: str, collection: str = COLLECTION, **kwargs) -> Document:
        document = project(raw_document(sha, filename, **kwargs))
        upsert(self.conn, document, collection)
        return document

    def test_the_schema_version_is_recorded(self) -> None:
        self.assertEqual(schema_version(self.conn), REGISTRY_SCHEMA_VERSION)

    def test_a_registry_is_created_where_it_is_asked_for(self) -> None:
        self.assertTrue(registry_path(self.tmp).exists())

    def test_a_document_round_trips(self) -> None:
        self._register("k1", "rehabGedung.pdf", number="42/X")
        stored = get(self.conn, "k1", COLLECTION)
        self.assertEqual(stored.filename, "rehabGedung.pdf")
        self.assertEqual(stored.contract_number, "42/X")
        self.assertEqual([p.organization for p in stored.parties], ["Dinas PUPR"])

    def test_registering_twice_does_not_duplicate(self) -> None:
        """`load.py` is resumable, so it will re-register a retried document."""
        self._register("k1", "rehabGedung.pdf")
        self._register("k1", "rehabGedung.pdf")
        self.assertEqual(count(self.conn, COLLECTION), 1)
        self.assertEqual(len(get(self.conn, "k1", COLLECTION).parties), 1,
                         "parties are replaced, not accumulated")

    def test_the_same_document_can_be_in_two_collections(self) -> None:
        """A re-embed under a new model must not disturb the old catalogue."""
        self._register("k1", "rehabGedung.pdf", collection=COLLECTION)
        self._register("k1", "rehabGedung.pdf", collection=OTHER_COLLECTION)
        self.assertEqual(count(self.conn, COLLECTION), 1)
        self.assertEqual(count(self.conn, OTHER_COLLECTION), 1)

    def test_names_returns_the_shape_the_scan_returns(self) -> None:
        """It stands in for `store.corpus_documents`, so it must match it."""
        self._register("k1", "rehabGedung.pdf")
        self._register("k2", "polres.pdf")
        self.assertEqual(names(self.conn, COLLECTION), {"k1": "rehabGedung.pdf", "k2": "polres.pdf"})

    def test_another_collection_is_not_listed(self) -> None:
        self._register("k1", "rehabGedung.pdf", collection=OTHER_COLLECTION)
        self.assertEqual(names(self.conn, COLLECTION), {})

    def test_a_missing_document_is_none_not_an_error(self) -> None:
        self.assertIsNone(get(self.conn, "nope", COLLECTION))


class StaleSchemaTests(unittest.TestCase):
    """A registry written by an older version of this module must cost a
    rebuild, never an exception: it is derived from the raw files, and
    everything that reads it is supposed to degrade to scanning."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = registry_path(self.tmp)

    def _old_registry(self, version: str = REGISTRY_SCHEMA_VERSION) -> None:
        """A `documents` table missing a column this module now writes."""
        connection = sqlite3.connect(self.path)
        connection.executescript(
            """CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
               CREATE TABLE documents (document_key TEXT, collection TEXT, filename TEXT);"""
        )
        connection.execute("INSERT INTO meta VALUES ('schema_version', ?)", (version,))
        connection.commit()
        connection.close()

    def test_an_older_table_is_reset_not_raised_on(self) -> None:
        """`CREATE TABLE IF NOT EXISTS` leaves it alone, so the first query fails."""
        self._old_registry()
        with self.assertLogs("retrieval.registry", level="WARNING"):
            connection = connect(self.path)
        self.addCleanup(connection.close)
        self.assertEqual(count(connection, COLLECTION), 0)
        upsert(connection, project(raw_document("k1", "rehabGedung.pdf")), COLLECTION)
        self.assertEqual(names(connection, COLLECTION), {"k1": "rehabGedung.pdf"})

    def test_a_different_version_is_reset_too(self) -> None:
        self._old_registry(version="0.0.1")
        with self.assertLogs("retrieval.registry", level="WARNING"):
            connection = connect(self.path)
        self.addCleanup(connection.close)
        self.assertEqual(schema_version(connection), REGISTRY_SCHEMA_VERSION)

    def test_a_new_file_is_not_reported_as_stale(self) -> None:
        with self.assertNoLogs("retrieval.registry", level="WARNING"):
            connection = connect(self.path)
        self.addCleanup(connection.close)

    def test_reopening_a_current_registry_keeps_its_contents(self) -> None:
        connection = connect(self.path)
        upsert(connection, project(raw_document("k1", "rehabGedung.pdf")), COLLECTION)
        connection.close()
        reopened = connect(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(names(reopened, COLLECTION), {"k1": "rehabGedung.pdf"})


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.conn = connect(registry_path(self.tmp))
        self.addCleanup(self.conn.close)
        for sha, filename, number in (
            ("k1", "Rancangan Kontrak.pdf", "08/PUPRPRKP-B.PNK/SP-PPK"),
            ("k2", "kontrakJasa.pdf", "99/JASA"),
            ("k3", "pembangunanRumah.pdf", "12/RUMAH"),
            ("k4", "pembangunanSayap.pdf", "13/SAYAP"),
            ("k5", "rehabGedung.pdf", "14/REHAB"),
        ):
            upsert(self.conn, project(raw_document(sha, filename, number=number)), COLLECTION)

    def test_a_full_name_finds_one_document(self) -> None:
        matches = search(self.conn, "rehabGedung", COLLECTION)
        self.assertEqual(matches.total, 1)
        self.assertEqual(matches.documents[0].filename, "rehabGedung.pdf")

    def test_a_partial_word_still_matches(self) -> None:
        """People type "pembangunan rumah" for `pembangunanRumah.pdf`."""
        self.assertGreaterEqual(search(self.conn, "pembangunan", COLLECTION).total, 2)

    def test_a_contract_number_is_searchable(self) -> None:
        """At scale the number distinguishes documents the filename cannot."""
        matches = search(self.conn, "08/PUPRPRKP-B.PNK/SP-PPK", COLLECTION)
        self.assertEqual(matches.documents[0].filename, "Rancangan Kontrak.pdf")

    def test_punctuation_is_not_read_as_a_query_operator(self) -> None:
        """A contract number is full of characters FTS5 treats as syntax."""
        for text in ("08/PUPR", "a-b", '"', "NOT OR AND", "*"):
            with self.subTest(text=text):
                search(self.conn, text, COLLECTION)  # must not raise

    def test_an_ambiguous_name_reports_a_count_not_a_catalogue(self) -> None:
        matches = search(self.conn, "kontrak", COLLECTION, limit=1)
        self.assertGreaterEqual(matches.total, 2)
        self.assertEqual(len(matches.documents), 1, "only the limit is returned")
        self.assertGreaterEqual(matches.truncated, 1, "the rest are counted, not listed")

    def test_no_match_is_empty_not_everything(self) -> None:
        """The failure the registry exists to fix: explaining a miss by
        printing the whole catalogue."""
        matches = search(self.conn, "anggaran2024", COLLECTION)
        self.assertEqual(matches.total, 0)
        self.assertEqual(matches.documents, [])

    def test_search_is_scoped_to_the_collection(self) -> None:
        self.assertEqual(search(self.conn, "rehabGedung", OTHER_COLLECTION).total, 0)


class FilenameMatchTests(unittest.TestCase):
    """What FTS5 cannot do, and why the normalised column exists."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.conn = connect(registry_path(self.tmp))
        self.addCleanup(self.conn.close)
        for sha, filename, name in (
            ("k1", "kontrakJasa.pdf", ""),                       # no contract name at all
            ("k2", "pembangunanRumah.pdf", "Pembangunan Rumah Transmigran"),
            ("k3", "pembangunanSayap.pdf", "Pembangunan Bangunan Sayap"),
            ("k4", "Rancangan Kontrak.pdf", "Peningkatan Jalan Mekar"),
        ):
            upsert(self.conn, project(raw_document(sha, filename, name=name)), COLLECTION)

    def test_a_camel_case_fragment_is_found(self) -> None:
        """`kontrakJasa` is one FTS token and the file has no contract name, so
        the full-text index cannot find "jasa" at all."""
        self.assertEqual(search(self.conn, "jasa", COLLECTION).total, 0, "FTS misses it")
        self.assertEqual(find_by_filename(self.conn, "jasa", COLLECTION).total, 1)

    def test_spacing_and_case_do_not_matter(self) -> None:
        matches = find_by_filename(self.conn, "pembangunan rumah", COLLECTION)
        self.assertEqual([d.filename for d in matches.documents], ["pembangunanRumah.pdf"])

    def test_an_ambiguous_fragment_returns_every_match_with_a_count(self) -> None:
        matches = find_by_filename(self.conn, "pembangunan", COLLECTION)
        self.assertEqual(matches.total, 2)

    def test_resolve_prefers_the_filename_then_falls_back_to_metadata(self) -> None:
        self.assertEqual(resolve(self.conn, "jasa", COLLECTION).documents[0].filename,
                         "kontrakJasa.pdf")
        # No filename holds "mekar"; the contract name does.
        self.assertEqual(resolve(self.conn, "mekar", COLLECTION).documents[0].filename,
                         "Rancangan Kontrak.pdf")

    def test_resolve_caps_what_it_returns_but_not_what_it_counts(self) -> None:
        matches = resolve(self.conn, "pembangunan", COLLECTION, limit=1)
        self.assertEqual(len(matches.documents), 1)
        self.assertEqual(matches.total, 2)
        self.assertEqual(matches.truncated, 1)

    def test_the_two_normalisers_agree(self) -> None:
        """`documents.normalize` applies the same rule; they must not drift
        while both exist."""
        from retrieval.documents import normalize as documents_normalize

        for text in ("pembangunanRumah", "Rancangan Kontrak", "08/PUPR-B.PNK", "a b  c", ""):
            with self.subTest(text=text):
                self.assertEqual(normalize_name(text), documents_normalize(text))


class FakeCollection:
    """The two calls `registry` makes on a collection."""

    def __init__(self, document_keys: list[str]) -> None:
        self._keys = document_keys

    def get(self, include=None):
        return {"metadatas": [{"document_key": key} for key in self._keys]}

    def count(self) -> int:
        return len(self._keys)


class CurrencyTests(unittest.TestCase):
    """Reading the registry instead of scanning is only safe while it accounts
    for every row the collection holds.

    An *incomplete* registry is the dangerous case: it answers with fewer
    documents than exist, so a question is scoped to a subset and answered
    confidently from it. A missing registry announces itself; a half-full one
    does not.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.conn = connect(registry_path(self.tmp))
        self.addCleanup(self.conn.close)

    def _register(self, key: str, filename: str, rows: int | None) -> None:
        document = project(raw_document(key, filename))
        document.row_count = rows
        upsert(self.conn, document, COLLECTION)

    def test_a_registry_matching_the_collection_is_current(self) -> None:
        self._register("k1", "a.pdf", 10)
        self._register("k2", "b.pdf", 5)
        self.assertTrue(is_current(self.conn, COLLECTION, 15))

    def test_a_registry_missing_a_document_is_not_current(self) -> None:
        """The silent-wrong-answer case this guard exists for."""
        self._register("k1", "a.pdf", 10)
        self.assertFalse(is_current(self.conn, COLLECTION, 15))

    def test_an_empty_registry_is_not_current(self) -> None:
        self.assertFalse(is_current(self.conn, COLLECTION, 15))

    def test_an_unverifiable_row_count_is_not_current(self) -> None:
        """A rebuild that could not read the collection leaves counts unset;
        an unverifiable total must not pass as a verified one."""
        self._register("k1", "a.pdf", 10)
        self._register("k2", "b.pdf", None)
        self.assertFalse(is_current(self.conn, COLLECTION, 15))

    def test_another_collections_rows_do_not_count(self) -> None:
        self._register("k1", "a.pdf", 10)
        document = project(raw_document("k2", "b.pdf"))
        document.row_count = 5
        upsert(self.conn, document, OTHER_COLLECTION)
        self.assertFalse(is_current(self.conn, COLLECTION, 15))
        self.assertTrue(is_current(self.conn, COLLECTION, 10))

    def test_row_counts_are_read_from_the_collection(self) -> None:
        counts = row_counts_from_collection(FakeCollection(["k1", "k1", "k2"]))
        self.assertEqual(counts, {"k1": 2, "k2": 1})

    def test_rows_without_a_document_key_are_ignored(self) -> None:
        """Rows predating document scoping carry no key."""
        collection = FakeCollection(["k1"])
        collection._keys = ["k1", None, ""]
        self.assertEqual(row_counts_from_collection(collection), {"k1": 1})


class RebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        for sha, filename in (("k1", "rehabGedung.pdf"), ("k2", "polres.pdf")):
            (self.raw / f"{filename[:-4]}_raw.json").write_text(
                json.dumps(raw_document(sha, filename)), encoding="utf-8")
        self.conn = connect(registry_path(self.tmp))
        self.addCleanup(self.conn.close)

    def test_rebuild_registers_every_raw_file(self) -> None:
        self.assertEqual(rebuild(self.conn, COLLECTION, self.raw), 2)
        self.assertEqual(count(self.conn, COLLECTION), 2)

    def test_rebuild_can_be_restricted_to_what_a_collection_holds(self) -> None:
        """A raw file on disk is not evidence that it was ever loaded."""
        self.assertEqual(rebuild(self.conn, COLLECTION, self.raw, only_keys={"k1"}), 1)
        self.assertEqual(list(names(self.conn, COLLECTION)), ["k1"])

    def test_rebuild_is_idempotent(self) -> None:
        rebuild(self.conn, COLLECTION, self.raw)
        rebuild(self.conn, COLLECTION, self.raw)
        self.assertEqual(count(self.conn, COLLECTION), 2)

    def test_an_unreadable_raw_file_is_skipped_not_fatal(self) -> None:
        """A registry is derived; one bad file must not cost the whole table."""
        (self.raw / "broken_raw.json").write_text("{not json", encoding="utf-8")
        with self.assertLogs("retrieval.registry", level="WARNING"):
            self.assertEqual(rebuild(self.conn, COLLECTION, self.raw), 2)

    def test_a_missing_raw_directory_is_not_fatal(self) -> None:
        with self.assertLogs("retrieval.registry", level="WARNING"):
            self.assertEqual(list(read_raw_documents(self.tmp / "nope")), [])


class RealCorpusTests(unittest.TestCase):
    """Against the actual raw extractions, so the projection is pinned to the
    shape `pipeline/` really writes rather than to the fixture above."""

    @classmethod
    def setUpClass(cls) -> None:
        from retrieval.registry import RAW_DIR
        if not RAW_DIR.is_dir() or not any(RAW_DIR.glob("*_raw.json")):
            raise unittest.SkipTest("no raw extractions on disk")
        cls.documents = [document for _, document in read_raw_documents()]

    def test_every_specimen_projects(self) -> None:
        self.assertGreaterEqual(len(self.documents), 1)
        for document in self.documents:
            with self.subTest(document=document.filename):
                self.assertTrue(document.document_key)
                self.assertTrue(document.filename)

    def test_document_keys_are_unique(self) -> None:
        keys = [d.document_key for d in self.documents]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_specimen_yields_a_nonsense_year(self) -> None:
        """2010 here would mean the account-code date won over the budget year."""
        for document in self.documents:
            with self.subTest(document=document.filename):
                self.assertIn(document.year, ("", *(str(y) for y in range(2015, 2031))))


if __name__ == "__main__":
    unittest.main()
