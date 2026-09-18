"""Settings, collection naming, and the safety defaults that live at the edge
of the retrieval layer: where the store is, when an API key is required, and
that Chroma telemetry is off.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from retrieval import config
from retrieval.config import (
    PROJECT_DIR,
    Settings,
    collection_name,
    load_settings,
    needs_api_key,
    resolve_db_path,
)

_ENV_KEYS = ("MISTRAL_API_KEY", "EMBEDDING_MODEL", "CHROMA_DB_PATH", "CHROMA_COLLECTION_PREFIX", "CHAT_MODEL")


class LoadSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Never read the developer's real .env; control the environment here.
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / ".env").write_text("", encoding="utf-8")
        self.enterContext(mock.patch.object(config, "ENV_PATH", self.tmp / ".env"))
        clean = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
        self.enterContext(mock.patch.dict(os.environ, clean, clear=True))

    def test_missing_key_is_refused_when_required(self) -> None:
        os.environ["EMBEDDING_MODEL"] = "mistral-embed"
        with self.assertRaises(SystemExit) as caught:
            load_settings()
        self.assertIn("MISTRAL_API_KEY is not set", str(caught.exception))

    def test_missing_key_is_fine_when_no_api_call_can_happen(self) -> None:
        os.environ["EMBEDDING_MODEL"] = "mistral-embed"
        settings = load_settings(require_api_key=False)
        self.assertEqual(settings.api_key, "")

    def test_embedding_model_must_be_pinned_even_without_a_key(self) -> None:
        """The model names the collection, so a silent default opens the wrong one."""
        with self.assertRaises(SystemExit) as caught:
            load_settings(require_api_key=False)
        self.assertIn("EMBEDDING_MODEL is not set", str(caught.exception))

    def test_collection_name_is_derived_not_configured(self) -> None:
        os.environ.update(EMBEDDING_MODEL="mistral-embed", MISTRAL_API_KEY="sk-test")
        self.assertEqual(load_settings().collection, collection_name("contracts", "mistral-embed"))

    def test_relative_db_path_resolves_from_the_project_not_the_cwd(self) -> None:
        """Resolving from the cwd creates a second, un-ignored store."""
        os.environ.update(EMBEDDING_MODEL="mistral-embed", CHROMA_DB_PATH="./chroma_data")
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(load_settings(require_api_key=False).db_path, (PROJECT_DIR / "chroma_data").resolve())

    def test_absolute_db_path_is_kept(self) -> None:
        self.assertEqual(resolve_db_path(str(self.tmp)), self.tmp.resolve())


class SmallRulesTests(unittest.TestCase):
    def test_which_runs_need_an_api_key(self) -> None:
        self.assertFalse(needs_api_key("bm25"))
        self.assertFalse(needs_api_key("bm25", "null"))
        self.assertTrue(needs_api_key("bm25", "mistral"))
        for retriever in ("dense", "brute", "hybrid", "hybrid-brute"):
            with self.subTest(retriever=retriever):
                self.assertTrue(needs_api_key(retriever))

    def test_redacted_key_never_exposes_more_than_a_prefix(self) -> None:
        secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
        settings = Settings(api_key=secret, model="m", batch_size=1, request_delay=0.0,
                            db_path=Path("."), collection="c")
        self.assertNotIn(secret[3:], settings.redacted_key)
        self.assertNotIn(secret[3:], repr(settings.redacted_key))
        self.assertEqual(Settings(api_key="", model="m", batch_size=1, request_delay=0.0,
                                  db_path=Path("."), collection="c").redacted_key, "MISSING")


class SecretsTests(unittest.TestCase):
    """"Secrets never touch git", pinned rather than re-checked by hand."""

    SECRET = "sk-abcdefghijklmnopqrstuvwxyz0123456789"

    def _settings(self) -> Settings:
        return Settings(api_key=self.SECRET, model="m", batch_size=1, request_delay=0.0,
                        db_path=Path("."), collection="c")

    def test_settings_repr_does_not_contain_the_key(self) -> None:
        """A formatted Settings object must not leak the key into a log."""
        self.assertNotIn(self.SECRET, repr(self._settings()))
        self.assertNotIn(self.SECRET, f"{self._settings()}")

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        if shutil.which("git") is None:
            self.skipTest("git not available")
        result = subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=60)
        if "not a git repository" in result.stderr:
            self.skipTest("not a git checkout")
        return result

    def test_env_files_and_chroma_stores_are_ignored_wherever_they_appear(self) -> None:
        for path in ("retrieval/.env", ".env", "../.env", "chroma_data/", "../chroma_data/", "retrieval/chroma_data/"):
            with self.subTest(path=path):
                self.assertEqual(self._git("check-ignore", "--no-index", "-q", path).returncode, 0,
                                 f"{path} would be committable")

    def test_env_template_stays_tracked(self) -> None:
        self.assertNotEqual(self._git("check-ignore", "--no-index", "-q", "retrieval/.env.example").returncode, 0)

    def test_env_template_carries_no_key(self) -> None:
        template = (PROJECT_DIR / "retrieval" / ".env.example").read_text(encoding="utf-8")
        key_lines = [line for line in template.splitlines() if line.startswith("MISTRAL_API_KEY")]
        self.assertEqual(key_lines, ["MISTRAL_API_KEY="])


class TelemetryTests(unittest.TestCase):
    def _client_telemetry(self, env: dict) -> str:
        """A fresh interpreter, because chromadb caches its settings per process."""
        code = (
            "import tempfile, retrieval, chromadb;"
            "print(chromadb.PersistentClient(path=tempfile.mkdtemp()).get_settings().anonymized_telemetry)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, cwd=PROJECT_DIR,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_importing_the_package_turns_chroma_telemetry_off(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANONYMIZED_TELEMETRY"}
        self.assertEqual(self._client_telemetry(env), "False")

    def test_an_empty_setting_is_treated_as_unset(self) -> None:
        self.assertEqual(self._client_telemetry(dict(os.environ, ANONYMIZED_TELEMETRY="")), "False")


if __name__ == "__main__":
    unittest.main()
