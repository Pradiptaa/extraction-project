"""`retrieval.store` — opening the collection, and resolving a document scope.

Scope resolution is the half of document scoping a user actually touches: they
type a name, and something has to turn it into `document_key`s. Its failure
modes are all quiet ones — scoping to the wrong contract, or silently to none —
so each is pinned here.

No API key and no real corpus: a temporary Chroma collection and a temporary
views directory are enough, because none of this embeds anything.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chromadb

from retrieval.store import (
    corpus_documents,
    describe_scope,
    document_names,
    resolve_scope,
)

KEY_A = "8489309d03524f6899afed3ad0f3659d2617e3285635e44507abca0958ac77a3"
KEY_B = "a7ed584e1c2b3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789"


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
        """`output/` is gitignored and regenerable, so names are a convenience.
        Scoping must keep working on a machine that has only the Chroma store."""
        self.assertEqual(document_names(self.tmp / "absent"), {})

    def test_documents_come_from_the_collection_not_the_views(self) -> None:
        """A view that was built but never loaded must not appear as an
        available scope — it names nothing that can actually be searched."""
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
        """The dangerous case: quietly picking one of two matches produces a
        confident answer about the wrong contract, which is precisely what
        scoping exists to prevent."""
        self._write_view("rehabGedungDuaLantai", "d" * 64)
        self.collection.add(ids=["d1"], embeddings=[[0.1, 0.9]], metadatas=[{"document_key": "d" * 64}],
                            documents=["empat"])
        with self.assertRaises(SystemExit) as caught:
            self._resolve("rehab")
        self.assertIn("ambiguous", str(caught.exception))

    def test_no_match_lists_what_is_available(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._resolve("kontrakJasa")
        message = str(caught.exception)
        self.assertIn("no document matches", message)
        self.assertIn("rehabGedung.pdf", message)

    def test_an_empty_spec_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._resolve("  ,  ")

    def test_a_collection_without_document_keys_says_so(self) -> None:
        """A collection predating scoping would otherwise fail every query with
        an empty result, which reads as catastrophic retrieval failure rather
        than as a collection that needs reloading."""
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
