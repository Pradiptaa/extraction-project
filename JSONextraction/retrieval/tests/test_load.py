"""Failure and resume behaviour for retrieval.load.

These use a fake embedder rather than the real API: the point is to prove what
happens when a batch fails partway, which is exactly the case a live run cannot
be relied upon to produce on demand (the real 429s during the first full load
were all absorbed by retry and never reached this path).

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
    """Stands in for retrieval.embed.Embedder. `fail_on_call` is 1-based so it
    reads the same way as the batch numbers in the log output."""

    instances: list["FakeEmbedder"] = []
    fail_on_call: int | None = None

    def __init__(self, api_key: str, model: str, request_delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.total_tokens = 0
        FakeEmbedder.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if FakeEmbedder.fail_on_call is not None and len(self.calls) == FakeEmbedder.fail_on_call:
            raise RuntimeError("simulated upstream failure")
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

    def _run(self):
        with mock.patch("retrieval.load.Embedder", FakeEmbedder):
            return run([self.view_path], self.settings)

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
        # The log has to carry enough to find the affected rows without a re-run.
        self.assertIn("first_id=", message)
        self.assertIn("node_ids=", message)

        # Only the successful batch is durable. If the failed rows were recorded
        # here, a resume would skip them forever and the collection would be
        # permanently short without ever reporting an error.
        self.assertEqual(self._collection().count(), 2)
        self.assertEqual(len(self._manifest_ids()), 2)

    def test_rerun_after_failure_embeds_only_the_missing_rows(self) -> None:
        FakeEmbedder.fail_on_call = 2
        self.assertEqual(self._run(), 1)
        first_pass = FakeEmbedder.instances[-1]

        FakeEmbedder.fail_on_call = None
        self.assertEqual(self._run(), 0, "the retry run should succeed")
        second_pass = FakeEmbedder.instances[-1]

        # The whole point of the manifest: the retry must not re-embed the rows
        # that already succeeded, or a failure late in a long run costs the
        # tokens for everything before it.
        self.assertEqual(len(second_pass.embedded_texts), 2)
        already_done = set(first_pass.calls[0])
        self.assertFalse(already_done & set(second_pass.embedded_texts))

        self.assertEqual(self._collection().count(), self.total_rows)
        self.assertEqual(len(self._manifest_ids()), self.total_rows)

    def test_rerun_after_clean_run_calls_the_api_zero_times(self) -> None:
        self.assertEqual(self._run(), 0)
        before = len(FakeEmbedder.instances)

        self.assertEqual(self._run(), 0)
        # No embedder is even constructed when there is nothing pending, so the
        # API client is never built and no key is ever used.
        self.assertEqual(len(FakeEmbedder.instances), before)

    def test_manifest_loss_falls_back_to_chroma(self) -> None:
        self.assertEqual(self._run(), 0)
        (self.settings.db_path / f"{self.settings.collection}.manifest.jsonl").unlink()
        before = len(FakeEmbedder.instances)

        self.assertEqual(self._run(), 0)
        # Chroma already holds every row, so a deleted manifest must not trigger
        # a full re-embed of work that is demonstrably already done.
        self.assertEqual(len(FakeEmbedder.instances), before)

    def test_corrupt_manifest_line_does_not_discard_earlier_progress(self) -> None:
        self.assertEqual(self._run(), 0)
        manifest = self.settings.db_path / f"{self.settings.collection}.manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as f:
            f.write('{"ids": ["truncated-')  # what a hard kill mid-write leaves

        self.assertEqual(len(self._manifest_ids()), self.total_rows)


if __name__ == "__main__":
    unittest.main()
