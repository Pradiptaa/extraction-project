"""Settings, collection naming, and the safety defaults that live at the edge
of the retrieval layer: where the store is, how a model name becomes a
collection name, and that Chroma telemetry is off.

This branch serves every model locally, so no run requires a credential — the
tests that once pinned that requirement now pin its absence.

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
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_HOST,
    PROJECT_DIR,
    Settings,
    collection_name,
    load_settings,
    model_slug,
    resolve_db_path,
)

_ENV_KEYS = ("MISTRAL_API_KEY", "EMBEDDING_MODEL", "CHROMA_DB_PATH", "CHROMA_COLLECTION_PREFIX",
             "CHAT_MODEL", "OLLAMA_HOST", "OLLAMA_NUM_CTX", "QUERY_EMBED_ON_CPU")


class LoadSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Never read the developer's real .env; control the environment here.
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / ".env").write_text("", encoding="utf-8")
        self.enterContext(mock.patch.object(config, "ENV_PATH", self.tmp / ".env"))
        clean = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
        self.enterContext(mock.patch.dict(os.environ, clean, clear=True))

    def test_settings_carry_no_credential_at_all(self) -> None:
        """Every model is served locally, so there is nothing to authorise —
        and nothing that could leak into a log."""
        os.environ["EMBEDDING_MODEL"] = "bge-m3"
        settings = load_settings()
        self.assertFalse(hasattr(settings, "api_key"))
        self.assertNotIn("key", repr(settings).lower())

    def test_embedding_model_must_be_pinned(self) -> None:
        """The model names the collection, so a silent default opens the wrong one."""
        with self.assertRaises(SystemExit) as caught:
            load_settings()
        self.assertIn("EMBEDDING_MODEL is not set", str(caught.exception))

    def test_collection_name_is_derived_not_configured(self) -> None:
        os.environ["EMBEDDING_MODEL"] = "bge-m3"
        self.assertEqual(load_settings().collection, collection_name("contracts", "bge-m3"))

    def test_ollama_host_and_context_have_defaults(self) -> None:
        os.environ["EMBEDDING_MODEL"] = "bge-m3"
        settings = load_settings()
        self.assertEqual(settings.host, DEFAULT_OLLAMA_HOST)
        self.assertEqual(settings.num_ctx, DEFAULT_NUM_CTX)

    def test_the_context_window_is_never_ollamas_silent_2048(self) -> None:
        """Ollama truncates past num_ctx without a word, which would drop
        retrieved clauses out of the prompt."""
        self.assertGreaterEqual(DEFAULT_NUM_CTX, 8192)

    def test_ollama_host_and_context_are_overridable(self) -> None:
        os.environ.update(EMBEDDING_MODEL="bge-m3", OLLAMA_HOST="http://box:99", OLLAMA_NUM_CTX="4096")
        settings = load_settings()
        self.assertEqual(settings.host, "http://box:99")
        self.assertEqual(settings.num_ctx, 4096)

    def test_a_query_is_embedded_off_the_gpu_by_default(self) -> None:
        """Sharing the GPU makes the embedding and chat models evict each other
        on every question — 17s of a 20s answer on a 6 GB card."""
        os.environ["EMBEDDING_MODEL"] = "bge-m3"
        settings = load_settings()
        self.assertTrue(settings.query_embed_on_cpu)
        self.assertEqual(settings.query_num_gpu, 0)

    def test_a_card_that_holds_both_models_can_turn_it_off(self) -> None:
        os.environ.update(EMBEDDING_MODEL="bge-m3", QUERY_EMBED_ON_CPU="false")
        settings = load_settings()
        self.assertFalse(settings.query_embed_on_cpu)
        self.assertIsNone(settings.query_num_gpu, "None lets Ollama choose")

    def test_an_empty_host_falls_back_rather_than_producing_a_bad_url(self) -> None:
        os.environ.update(EMBEDDING_MODEL="bge-m3", OLLAMA_HOST="")
        self.assertEqual(load_settings().host, DEFAULT_OLLAMA_HOST)

    def test_relative_db_path_resolves_from_the_project_not_the_cwd(self) -> None:
        """Resolving from the cwd creates a second, un-ignored store."""
        os.environ.update(EMBEDDING_MODEL="bge-m3", CHROMA_DB_PATH="./chroma_data")
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(load_settings().db_path, (PROJECT_DIR / "chroma_data").resolve())

    def test_absolute_db_path_is_kept(self) -> None:
        self.assertEqual(resolve_db_path(str(self.tmp)), self.tmp.resolve())


class ModelSlugTests(unittest.TestCase):
    """A model name has to survive into a Chroma collection name without
    colliding with another model's."""

    def test_a_name_chroma_already_accepts_is_untouched(self) -> None:
        """Collections written before slugging existed must keep their names."""
        for model in ("mistral-embed", "bge-m3", "nomic-embed-text"):
            with self.subTest(model=model):
                self.assertEqual(model_slug(model), model)

    def test_an_ollama_tag_loses_the_character_chroma_rejects(self) -> None:
        self.assertNotIn(":", model_slug("bge-m3:latest"))

    def test_a_hugging_face_id_loses_its_slash(self) -> None:
        self.assertNotIn("/", model_slug("BAAI/bge-m3"))

    def test_the_field_separator_can_never_come_from_a_model_name(self) -> None:
        """`__` separates the fields of a collection name; one arriving from a
        model name would make the name ambiguous to parse."""
        self.assertNotIn("__", model_slug("bge__m3"))

    def test_names_that_differ_only_in_punctuation_get_different_slugs(self) -> None:
        """Substitution alone maps `bge:m3` and `bge/m3` onto one collection,
        which would silently mix two models' vectors."""
        self.assertNotEqual(model_slug("bge:m3"), model_slug("bge/m3"))

    def test_a_slug_is_legal_in_a_chroma_collection_name(self) -> None:
        for model in ("bge-m3:latest", "BAAI/bge-m3", "qwen2.5:3b-instruct", "a.b:c/d"):
            with self.subTest(model=model):
                name = collection_name("contracts", model)
                self.assertRegex(name, r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")

    def test_the_slug_a_collection_carries_is_the_one_settings_reports(self) -> None:
        """`load.py` matches `--reuse-from` against this; the two must agree."""
        settings = Settings(model="bge-m3:latest", batch_size=1,
                            db_path=Path("."), collection=collection_name("c", "bge-m3:latest"))
        self.assertIn(f"__{settings.model_slug}__", settings.collection)


class SecretsTests(unittest.TestCase):
    """"Secrets never touch git", pinned rather than re-checked by hand.

    This branch holds no credential, so the leak tests are gone with the field.
    What remains guards the store and any `.env` a developer still keeps here.
    """

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

    def test_env_template_asks_for_no_credential(self) -> None:
        """A template with a key slot invites someone to fill one in and commit it."""
        template = (PROJECT_DIR / "retrieval" / ".env.example").read_text(encoding="utf-8")
        for line in template.splitlines():
            name = line.split("=", 1)[0].strip()
            if line.startswith("#") or "=" not in line:
                continue
            self.assertNotRegex(name, r"(?i)(api_key|token|secret|password)")


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
