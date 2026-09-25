"""`retrieval.store` — opening the collection, and resolving a document scope.
Scope resolution's failure modes are all quiet ones (the wrong contract, or
silently none), so each is pinned here. Nothing embeds, so a temporary Chroma
collection and views directory are enough.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chromadb

from retrieval import registry, store
from retrieval.config import Settings
from retrieval.store import (
    corpus_documents,
    describe_scope,
    document_names,
    resolve_scope,
)

KEY_A = "8489309d03524f6899afed3ad0f3659d2617e3285635e44507abca0958ac77a3"
KEY_B = "a7ed584e1c2b3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789"


class RegistryBackedTests(unittest.TestCase):
    """`corpus_documents` prefers the indexed catalogue and falls back to the
    scan.

    The fallback is the whole safety story: a registry that is absent, stale or
    *incomplete* must cost speed and nothing else. An incomplete one is the
    dangerous case — it would hand back fewer documents than the collection
    holds, and a question would be scoped to a subset and answered confidently
    from it.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.views = self.tmp / "embedding"
        self.views.mkdir()
        for stem, key in (("Rancangan Kontrak", KEY_A), ("rehabGedung", KEY_B)):
            (self.views / f"{stem}_embedding_view.json").write_text(
                json.dumps({"source": {"file": f"{stem}.pdf", "raw_extraction_sha256": key}}),
                encoding="utf-8")

        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        for stem, key in (("Rancangan Kontrak", KEY_A), ("rehabGedung", KEY_B)):
            (self.raw / f"{stem}_raw.json").write_text(json.dumps({
                "source": {"file": f"{stem}.pdf", "sha256": key},
                "core": {},
            }), encoding="utf-8")
        self.enterContext(mock.patch.object(registry, "RAW_DIR", self.raw))
        self.enterContext(mock.patch.object(store, "VIEWS_DIR", self.views))

        self.db = self.tmp / "chroma"
        client = chromadb.PersistentClient(path=str(self.db))
        self.collection = client.get_or_create_collection("scope_rows")
        self.collection.add(
            ids=["a1", "a2", "b1"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            metadatas=[{"document_key": KEY_A}, {"document_key": KEY_A}, {"document_key": KEY_B}],
            documents=["satu", "dua", "tiga"],
        )
        self.settings = Settings(model="fake-model", batch_size=2,
                                 db_path=self.db, collection="scope_rows")

    def _build_registry(self, row_counts=None) -> None:
        connection = registry.connect(registry.registry_path(self.db))
        registry.rebuild(connection, "scope_rows", self.raw,
                         row_counts=row_counts if row_counts is not None
                         else {KEY_A: 2, KEY_B: 1})
        connection.close()

    def test_without_settings_nothing_but_the_scan_is_used(self) -> None:
        """Callers that never heard of the registry keep their behaviour."""
        self._build_registry()
        with mock.patch.object(store, "_registry_is_current") as registered:
            corpus_documents(self.collection, self.views)
        registered.assert_not_called()

    def test_the_registry_and_the_scan_agree(self) -> None:
        """The property every caller depends on: same question, same answer,
        whichever path produced it."""
        self._build_registry()
        self.assertEqual(
            corpus_documents(self.collection, self.views, self.settings),
            store.scan_corpus_documents(self.collection, self.views),
        )

    def test_the_registry_is_used_when_it_is_current(self) -> None:
        self._build_registry()
        with mock.patch.object(store, "scan_corpus_documents") as scan:
            result = corpus_documents(self.collection, self.views, self.settings)
        scan.assert_not_called()
        self.assertEqual(set(result), {KEY_A, KEY_B})

    def test_an_absent_registry_falls_back_silently(self) -> None:
        """Not having built one yet is normal, not a fault worth warning about."""
        self.assertEqual(set(corpus_documents(self.collection, self.views, self.settings)),
                         {KEY_A, KEY_B})

    def test_an_absent_registry_is_not_created_by_a_reader(self) -> None:
        corpus_documents(self.collection, self.views, self.settings)
        self.assertFalse(registry.registry_path(self.db).exists(),
                         "a question must not write a catalogue")

    def test_an_incomplete_registry_falls_back_and_says_so(self) -> None:
        """The silent-wrong-answer case: fewer documents than the collection holds."""
        self._build_registry(row_counts={KEY_A: 2})       # KEY_B never registered
        connection = registry.connect(registry.registry_path(self.db))
        connection.execute("DELETE FROM documents WHERE document_key = ?", (KEY_B,))
        connection.commit()
        connection.close()
        with self.assertLogs("retrieval.store", level="WARNING") as captured:
            result = corpus_documents(self.collection, self.views, self.settings)
        self.assertEqual(set(result), {KEY_A, KEY_B}, "the scan answered")
        self.assertIn("registry rebuild", "\n".join(captured.output))

    def test_a_registry_with_unverifiable_counts_falls_back(self) -> None:
        self._build_registry(row_counts={})               # no counts at all
        with self.assertLogs("retrieval.store", level="WARNING"):
            self.assertEqual(set(corpus_documents(self.collection, self.views, self.settings)),
                             {KEY_A, KEY_B})

    def test_a_registry_from_an_older_schema_falls_back(self) -> None:
        path = registry.registry_path(self.db)
        path.parent.mkdir(parents=True, exist_ok=True)
        stale = sqlite3.connect(path)
        stale.executescript(
            """CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
               CREATE TABLE documents (document_key TEXT, collection TEXT, filename TEXT);"""
        )
        stale.execute("INSERT INTO meta VALUES ('schema_version', '0.0.1')")
        stale.commit()
        stale.close()
        with self.assertLogs("retrieval.registry", level="WARNING"):
            result = corpus_documents(self.collection, self.views, self.settings)
        self.assertEqual(set(result), {KEY_A, KEY_B})

    def test_a_reader_does_not_reset_an_older_registry(self) -> None:
        """`connect` resets a stale schema; a question must not. Whoever runs
        the rebuild should find what was there, not an emptied file."""
        path = registry.registry_path(self.db)
        path.parent.mkdir(parents=True, exist_ok=True)
        stale = sqlite3.connect(path)
        stale.executescript("CREATE TABLE documents (whatever TEXT);")
        stale.commit()
        stale.close()
        before = path.read_bytes()
        with self.assertLogs("retrieval.registry", level="WARNING"):
            corpus_documents(self.collection, self.views, self.settings)
        self.assertEqual(path.read_bytes(), before, "the reader left it alone")

    def test_scope_resolution_works_through_the_registry(self) -> None:
        self._build_registry()
        self.assertEqual(resolve_scope("rehab", self.collection, self.views, self.settings),
                         {KEY_B: "rehabGedung.pdf"})

    # -- a registry that fails must cost a scan, never a traceback ----------

    def test_a_zero_byte_registry_falls_back(self) -> None:
        """What an interrupted `connect` leaves behind. It opens cleanly and
        then has no tables, so it used to fail on the first query."""
        path = registry.registry_path(self.db)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        with self.assertLogs("retrieval.registry", level="WARNING"):
            self.assertEqual(set(corpus_documents(self.collection, self.views, self.settings)),
                             {KEY_A, KEY_B})
        with self.assertLogs("retrieval.registry", level="WARNING"):
            matches = store.document_resolver(self.collection, self.views, self.settings)("rehab")
        self.assertEqual([d.filename for d in matches.documents], ["rehabGedung.pdf"])

    def test_a_read_that_fails_after_opening_falls_back(self) -> None:
        """Opening is not the only thing that fails: a dropped table, a corrupt
        page or a lock all surface on a later query."""
        self._build_registry()
        connection = registry.connect(registry.registry_path(self.db))
        connection.execute("DROP TABLE documents_fts")
        connection.commit()
        connection.close()
        resolve = store.document_resolver(self.collection, self.views, self.settings)
        with self.assertLogs("retrieval.registry", level="WARNING"):
            matches = resolve("anggaran")          # no filename match: reaches the FTS search
        self.assertEqual(matches.total, 0, "answered by the scan, not by an exception")

    def test_a_failing_registry_does_not_report_a_real_file_as_missing(self) -> None:
        """Answering a failed read with "no matches" would make a strong cue
        say the file does not exist. The scan answers instead."""
        self._build_registry()
        with mock.patch.object(registry, "resolve", side_effect=sqlite3.OperationalError("locked")):
            resolve = store.document_resolver(self.collection, self.views, self.settings)
            with self.assertLogs("retrieval.registry", level="WARNING"):
                matches = resolve("rehab")
        self.assertEqual([d.filename for d in matches.documents], ["rehabGedung.pdf"])

    def test_the_scan_behind_a_resolver_is_built_at_most_once(self) -> None:
        resolve = store.document_resolver(self.collection, self.views, self.settings)
        with mock.patch.object(store, "scan_corpus_documents",
                               wraps=store.scan_corpus_documents) as scan:
            resolve("rehab")
            resolve("rancangan")
        self.assertEqual(scan.call_count, 1)

    def test_a_stale_registry_warns_once_per_resolver(self) -> None:
        """The currency check used to run twice on the fallback path."""
        self._build_registry(row_counts={KEY_A: 99, KEY_B: 99})
        with self.assertLogs("retrieval.store", level="WARNING") as captured:
            store.document_resolver(self.collection, self.views, self.settings)("rehab")
        self.assertEqual(len(captured.output), 1)


class ScopeResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.views = self.tmp / "embedding"
        self.views.mkdir()
        self._write_view("Rancangan Kontrak", KEY_A)
        self._write_view("rehabGedung", KEY_B)

        client = chromadb.PersistentClient(path=str(self.tmp / "chroma"))
        self.collection = client.get_or_create_collection("scope_rows")
        self.collection.add(
            ids=["a1", "a2", "b1"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            metadatas=[{"document_key": KEY_A}, {"document_key": KEY_A}, {"document_key": KEY_B}],
            documents=["satu", "dua", "tiga"],
        )

    def _write_view(self, stem: str, key: str) -> None:
        path = self.views / f"{stem}_embedding_view.json"
        path.write_text(
            json.dumps({"source": {"file": f"{stem}.pdf", "raw_extraction_sha256": key}}),
            encoding="utf-8",
        )

    def _resolve(self, spec: str) -> dict[str, str]:
        return resolve_scope(spec, self.collection, self.views)

    # -- names -------------------------------------------------------------

    def test_names_are_read_from_the_embedding_views(self) -> None:
        self.assertEqual(document_names(self.views), {KEY_A: "Rancangan Kontrak.pdf", KEY_B: "rehabGedung.pdf"})

    def test_a_missing_views_directory_is_not_an_error(self) -> None:
        """Names are a convenience; scoping must work with only the Chroma store."""
        self.assertEqual(document_names(self.tmp / "absent"), {})

    def test_documents_come_from_the_collection_not_the_views(self) -> None:
        """A view built but never loaded names nothing that can be searched."""
        self._write_view("neverLoaded", "c" * 64)
        self.assertEqual(set(corpus_documents(self.collection, self.views)), {KEY_A, KEY_B})

    def test_documents_without_a_view_still_resolve_by_key(self) -> None:
        available = corpus_documents(self.collection, self.tmp / "absent")
        self.assertEqual(set(available), {KEY_A, KEY_B})
        self.assertEqual(resolve_scope(KEY_B[:8], self.collection, self.tmp / "absent"), {KEY_B: ""})

    # -- matching ----------------------------------------------------------

    def test_filename_substring_matches_case_insensitively(self) -> None:
        self.assertEqual(self._resolve("rehabgedung"), {KEY_B: "rehabGedung.pdf"})

    def test_document_key_prefix_matches(self) -> None:
        self.assertEqual(self._resolve(KEY_A[:8]), {KEY_A: "Rancangan Kontrak.pdf"})

    def test_several_documents_can_be_named_at_once(self) -> None:
        self.assertEqual(set(self._resolve(f"rehab, {KEY_A[:8]}")), {KEY_A, KEY_B})

    # -- refusals ----------------------------------------------------------

    def test_an_ambiguous_term_is_refused_rather_than_guessed(self) -> None:
        """Picking one of two matches gives a confident answer about the wrong
        contract, which is what scoping exists to prevent."""
        self._write_view("rehabGedungDuaLantai", "d" * 64)
        self.collection.add(ids=["d1"], embeddings=[[0.1, 0.9]], metadatas=[{"document_key": "d" * 64}],
                            documents=["empat"])
        with self.assertRaises(SystemExit) as caught:
            self._resolve("rehab")
        self.assertIn("ambiguous", str(caught.exception))

    def test_no_match_points_to_the_listing_instead_of_printing_it(self) -> None:
        """Printing every document explained a miss while there were six. At a
        thousand it is not an explanation, so the reply says where to look."""
        with self.assertRaises(SystemExit) as caught:
            self._resolve("kontrakJasa")
        message = str(caught.exception)
        self.assertIn("no document matches", message)
        self.assertNotIn("rehabGedung.pdf", message)
        self.assertIn("--list-documents", message)

    def test_a_spaced_name_matches_a_camel_case_file(self) -> None:
        """`--document` matches the way a name in a question does."""
        self.assertEqual(self._resolve("rehab gedung"), {KEY_B: "rehabGedung.pdf"})

    def test_the_extension_may_be_typed(self) -> None:
        self.assertEqual(self._resolve("rehabGedung.pdf"), {KEY_B: "rehabGedung.pdf"})

    def test_an_empty_spec_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._resolve("  ,  ")

    def test_a_collection_without_document_keys_says_so(self) -> None:
        """Otherwise it reads as catastrophic retrieval failure rather than a
        collection that needs reloading."""
        client = chromadb.PersistentClient(path=str(self.tmp / "chroma"))
        bare = client.get_or_create_collection("bare_rows")
        bare.add(ids=["x"], embeddings=[[1.0, 0.0]], metadatas=[{"label": "1"}], documents=["satu"])
        with self.assertRaises(SystemExit) as caught:
            resolve_scope("anything", bare, self.views)
        self.assertIn("document scoping", str(caught.exception))

    # -- description -------------------------------------------------------

    def test_describe_prefers_names_and_falls_back_to_keys(self) -> None:
        self.assertEqual(describe_scope({KEY_A: "Rancangan Kontrak.pdf"}), "Rancangan Kontrak.pdf")
        self.assertEqual(describe_scope({KEY_A: ""}), KEY_A[:12])


if __name__ == "__main__":
    unittest.main()
