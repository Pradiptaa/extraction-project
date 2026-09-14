"""Unit tests for retrieval.reindex.

No API key and no embedding calls — the whole point of the module is that it
moves vectors that already exist.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import chromadb

from retrieval.config import INDEX_METADATA, collection_name
from retrieval.reindex import copy_collection, verify

ROWS = [
    (f"id_{n}", [float(n % 7), float(n % 5), float(n % 3), 1.0], {"label": str(n)}, f"text {n % 11}")
    for n in range(40)
]


class ReindexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.client = chromadb.PersistentClient(path=str(self.tmp / "chroma"))
        self.source = self.client.get_or_create_collection(
            "old_name", metadata={"hnsw:space": "cosine"}
        )
        self.source.add(
            ids=[r[0] for r in ROWS],
            embeddings=[r[1] for r in ROWS],
            metadatas=[r[2] for r in ROWS],
            documents=[r[3] for r in ROWS],
        )

    def test_every_row_is_copied_with_its_documents_and_metadata(self) -> None:
        target = copy_collection(self.client, "old_name", "new_name")
        self.assertEqual(target.count(), len(ROWS))

        got = target.get(include=["documents", "metadatas", "embeddings"])
        by_id = dict(zip(got["ids"], got["documents"]))
        self.assertEqual(by_id["id_7"], "text 7")
        self.assertEqual(len(got["embeddings"][0]), 4)

    def test_source_is_left_untouched(self) -> None:
        """The old index is the fallback if a rebuild turns out worse, so it
        must survive intact."""
        copy_collection(self.client, "old_name", "new_name")
        self.assertEqual(self.source.count(), len(ROWS))

    def test_target_gets_the_configured_index_parameters(self) -> None:
        copy_collection(self.client, "old_name", "new_name")
        metadata = self.client.get_collection("new_name").metadata
        for key, value in INDEX_METADATA.items():
            self.assertEqual(metadata.get(key), value, f"{key} was not applied")

    def test_rerunning_is_a_no_op_rather_than_a_duplicate_load(self) -> None:
        copy_collection(self.client, "old_name", "new_name")
        again = copy_collection(self.client, "old_name", "new_name")
        self.assertEqual(again.count(), len(ROWS))

    def test_partial_target_is_completed_not_skipped(self) -> None:
        """A rebuild killed halfway leaves a short collection. Re-running must
        finish it, not accept it as done."""
        partial = self.client.get_or_create_collection("new_name", metadata=INDEX_METADATA)
        partial.add(
            ids=[r[0] for r in ROWS[:5]],
            embeddings=[r[1] for r in ROWS[:5]],
            metadatas=[r[2] for r in ROWS[:5]],
            documents=[r[3] for r in ROWS[:5]],
        )
        target = copy_collection(self.client, "old_name", "new_name")
        self.assertEqual(target.count(), len(ROWS))

    def test_dry_run_writes_nothing(self) -> None:
        self.assertIsNone(copy_collection(self.client, "old_name", "new_name", dry_run=True))
        self.assertNotIn("new_name", [c.name for c in self.client.list_collections()])

    def test_empty_source_is_refused(self) -> None:
        self.client.get_or_create_collection("empty", metadata=INDEX_METADATA)
        with self.assertRaises(SystemExit):
            copy_collection(self.client, "empty", "new_name")

    def test_verify_reports_full_recall_on_a_small_collection(self) -> None:
        target = copy_collection(self.client, "old_name", "new_name")
        self.assertGreaterEqual(verify(target, sample=8, k=3), 0.999)


class CollectionNameTests(unittest.TestCase):
    def test_index_tag_is_part_of_the_name(self) -> None:
        """HNSW parameters cannot be changed in place, so a parameter change has
        to land in a differently-named collection."""
        name = collection_name("contracts", "mistral-embed", "2.0.0", index_tag="hnsw-m64ef400")
        self.assertEqual(name, "contracts__mistral-embed__v2_0_0__hnsw-m64ef400")

    def test_untagged_name_is_still_addressable(self) -> None:
        """Collections written before index tagging existed must stay reachable,
        or the pre-rebuild index cannot be re-checked."""
        self.assertEqual(
            collection_name("contracts", "mistral-embed", "2.0.0", index_tag=None),
            "contracts__mistral-embed__v2_0_0",
        )


if __name__ == "__main__":
    unittest.main()
