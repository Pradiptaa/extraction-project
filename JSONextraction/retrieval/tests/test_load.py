"""Failure and resume behaviour for retrieval.load. Uses a fake embedder, since
a live run cannot be made to fail partway on demand.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from retrieval.build_embedding_view import build_embedding_view
from retrieval.config import Settings, collection_name
from retrieval.load import load_manifest, run

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "sample_raw_extraction.json"
DIM = 8


class FakeEmbedder:
    """Stands in for retrieval.embed.Embedder. `fail_on_call` is 1-based, like
    the batch numbers in the log output."""

    instances: list["FakeEmbedder"] = []
    fail_on_call: int | None = None
    failure: type[BaseException] = RuntimeError

    def __init__(self, api_key: str, model: str, request_delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.total_tokens = 0
        FakeEmbedder.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if FakeEmbedder.fail_on_call is not None and len(self.calls) == FakeEmbedder.fail_on_call:
            raise FakeEmbedder.failure("simulated upstream failure")
        self.total_tokens += sum(len(t) for t in texts)
        return [[float(len(t) % 10)] * DIM for t in texts]

    @property
    def embedded_texts(self) -> list[str]:
        return [t for call in self.calls for t in call]


class LoadFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        view = build_embedding_view(document)
        self.view_path = self.tmp / "sample_embedding_view.json"
        self.view_path.write_text(json.dumps(view), encoding="utf-8")
        self.total_rows = view["node_count"]

        self.settings = Settings(
            api_key="fake", model="fake-model", batch_size=2, request_delay=0.0,
            db_path=self.tmp / "chroma", collection=collection_name("test", "fake-model"),
        )
        FakeEmbedder.instances = []
        FakeEmbedder.fail_on_call = None
        FakeEmbedder.failure = RuntimeError

    def _run(self, reuse_from=None, settings=None):
        with mock.patch("retrieval.load.Embedder", FakeEmbedder):
            return run([self.view_path], settings or self.settings, reuse_from=reuse_from)

    def _collection(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.settings.db_path))
        return client.get_collection(self.settings.collection)

    def _manifest_ids(self) -> set[str]:
        return load_manifest(self.settings.db_path / f"{self.settings.collection}.manifest.jsonl")

    def test_clean_run_loads_every_row(self) -> None:
        self.assertEqual(self._run(), 0)
        self.assertEqual(self._collection().count(), self.total_rows)
        self.assertEqual(len(self._manifest_ids()), self.total_rows)

    def test_failed_batch_is_reported_and_not_recorded_as_done(self) -> None:
        # 4 fixture nodes at batch_size=2 means 2 batches; fail the second.
        FakeEmbedder.fail_on_call = 2
        with self.assertLogs("retrieval.load", level="ERROR") as captured:
            exit_code = self._run()

        self.assertEqual(exit_code, 1, "a failed batch must produce a non-zero exit code")

        message = "\n".join(captured.output)
        self.assertIn("batch 2/2 FAILED", message)
        self.assertIn("RuntimeError", message)
        # The log must locate the affected rows without a re-run.
        self.assertIn("first_id=", message)
        self.assertIn("node_ids=", message)

        # Recording the failed rows would make a resume skip them forever.
        self.assertEqual(self._collection().count(), 2)
        self.assertEqual(len(self._manifest_ids()), 2)

    def test_rerun_after_failure_embeds_only_the_missing_rows(self) -> None:
        FakeEmbedder.fail_on_call = 2
        self.assertEqual(self._run(), 1)
        first_pass = FakeEmbedder.instances[-1]

        FakeEmbedder.fail_on_call = None
        self.assertEqual(self._run(), 0, "the retry run should succeed")
        second_pass = FakeEmbedder.instances[-1]

        # A retry must not re-embed the rows that already succeeded.
        self.assertEqual(len(second_pass.embedded_texts), 2)
        already_done = set(first_pass.calls[0])
        self.assertFalse(already_done & set(second_pass.embedded_texts))

        self.assertEqual(self._collection().count(), self.total_rows)
        self.assertEqual(len(self._manifest_ids()), self.total_rows)

    def test_rerun_after_clean_run_calls_the_api_zero_times(self) -> None:
        self.assertEqual(self._run(), 0)
        before = len(FakeEmbedder.instances)

        self.assertEqual(self._run(), 0)
        # With nothing pending no embedder is constructed, so no key is used.
        self.assertEqual(len(FakeEmbedder.instances), before)

    def test_manifest_loss_falls_back_to_chroma(self) -> None:
        self.assertEqual(self._run(), 0)
        (self.settings.db_path / f"{self.settings.collection}.manifest.jsonl").unlink()
        before = len(FakeEmbedder.instances)

        self.assertEqual(self._run(), 0)
        # Chroma holds every row, so a deleted manifest must not re-embed them.
        self.assertEqual(len(FakeEmbedder.instances), before)

    def test_stale_manifest_does_not_hide_rows_missing_from_chroma(self) -> None:
        """The manifest survives a deleted collection, so Chroma must win and
        the rows be re-embedded rather than skipped forever."""
        self.assertEqual(self._run(), 0)
        import chromadb

        chromadb.PersistentClient(path=str(self.settings.db_path)).delete_collection(self.settings.collection)
        self.assertEqual(len(self._manifest_ids()), self.total_rows, "precondition: manifest survives")

        with self.assertLogs("retrieval.load", level="WARNING") as captured:
            self.assertEqual(self._run(), 0)
        self.assertIn("does not hold them", "\n".join(captured.output))
        self.assertEqual(len(FakeEmbedder.instances[-1].embedded_texts), self.total_rows)
        self.assertEqual(self._collection().count(), self.total_rows)

    def test_non_retryable_failure_stops_the_run(self) -> None:
        """A bad key or dimension change fails every batch identically, so the
        run must stop at the first rather than repeat the error N times."""
        FakeEmbedder.fail_on_call = 1  # RuntimeError: not transient
        with self.assertLogs("retrieval.load", level="ERROR") as captured:
            self.assertEqual(self._run(), 1)

        self.assertEqual(len(FakeEmbedder.instances[-1].calls), 1, "no batch after the fatal one")
        message = "\n".join(captured.output)
        self.assertIn("aborting", message)
        self.assertIn("1 remaining batch", message)
        self.assertEqual(self._collection().count(), 0)

    def test_transient_failure_does_not_stop_the_run(self) -> None:
        """The remaining batches are still worth attempting; the failed one is
        left for the next run."""
        FakeEmbedder.fail_on_call = 1
        FakeEmbedder.failure = TimeoutError
        self.assertEqual(self._run(), 1)

        self.assertEqual(len(FakeEmbedder.instances[-1].calls), 2, "batch 2 still attempted")
        self.assertEqual(self._collection().count(), 2)

    def _new_schema_settings(self, model: str = "fake-model") -> Settings:
        from dataclasses import replace

        return replace(self.settings, model=model, collection=collection_name("test", model, schema_version="9.9.9"))

    def test_reuse_copies_stored_vectors_and_embeds_only_new_rows(self) -> None:
        """A schema bump that keeps ids stable must not re-spend tokens: only
        the changed node is embedded, and its old version is not carried over."""
        self.assertEqual(self._run(), 0)
        old_ids = set(self._collection().get(include=[])["ids"])

        document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        document["structure"][1]["text_raw"] = "text changed by a pipeline fix"
        self.view_path.write_text(json.dumps(build_embedding_view(document)), encoding="utf-8")
        target = self._new_schema_settings()

        self.assertEqual(self._run(reuse_from=self.settings.collection, settings=target), 0)
        self.assertEqual(len(FakeEmbedder.instances[-1].embedded_texts), 1)

        import chromadb

        client = chromadb.PersistentClient(path=str(target.db_path))
        new = client.get_collection(target.collection).get(include=["embeddings"])
        old = self._collection().get(include=["embeddings"])
        old_vectors = dict(zip(old["ids"], [list(v) for v in old["embeddings"]]))
        new_ids = set(new["ids"])
        self.assertEqual(len(new_ids), self.total_rows)
        self.assertEqual(len(new_ids - old_ids), 1, "exactly one row is new")
        self.assertEqual(len(old_ids - new_ids), 1, "the stale row was not copied")
        for row_id, vector in zip(new["ids"], new["embeddings"]):
            if row_id in old_vectors:
                self.assertEqual(list(vector), old_vectors[row_id])

    def test_reuse_refuses_a_collection_built_by_another_model(self) -> None:
        self.assertEqual(self._run(), 0)
        target = self._new_schema_settings(model="other-model")
        with self.assertRaises(SystemExit) as caught:
            self._run(reuse_from=self.settings.collection, settings=target)
        self.assertIn("refusing to mix", str(caught.exception))

    def test_reuse_dry_run_writes_nothing(self) -> None:
        self.assertEqual(self._run(), 0)
        target = self._new_schema_settings()
        with mock.patch("retrieval.load.Embedder", FakeEmbedder):
            self.assertEqual(run([self.view_path], target, dry_run=True, reuse_from=self.settings.collection), 0)
        import chromadb

        self.assertEqual(chromadb.PersistentClient(path=str(target.db_path)).get_collection(target.collection).count(), 0)

    def test_corrupt_manifest_line_does_not_discard_earlier_progress(self) -> None:
        self.assertEqual(self._run(), 0)
        manifest = self.settings.db_path / f"{self.settings.collection}.manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as f:
            f.write('{"ids": ["truncated-')  # what a hard kill mid-write leaves

        self.assertEqual(len(self._manifest_ids()), self.total_rows)


if __name__ == "__main__":
    unittest.main()
