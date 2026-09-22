"""`vocabulary` — the single source of the domain's words, and the rule that
every layer reads the same one.

The point of Phase 2 is that adding a contract family is a JSON file. These
tests fail when a term drifts back into code, or when one layer learns a subtype
the others cannot name.
"""
from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

from vocabulary import (
    BASE_VOCABULARY_ID,
    PROFILE_DIR,
    for_all_profiles,
    for_profile,
    for_profile_id,
    known_profiles,
)

REPO = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO / "pipeline"
RETRIEVAL_DIR = REPO / "retrieval"


class VocabularyLoadingTests(unittest.TestCase):
    def test_base_layer_is_not_a_selectable_profile(self) -> None:
        base = json.loads((PROFILE_DIR / f"{BASE_VOCABULARY_ID}.json").read_text(encoding="utf-8"))
        self.assertNotIn("profile_id", base)
        self.assertNotIn(BASE_VOCABULARY_ID, [p["profile_id"] for p in known_profiles()])

    def test_profile_inherits_the_base_layer(self) -> None:
        perpres = for_profile_id("perpres16_konstruksi_v1")
        self.assertIn("Nomor Kontrak", perpres.labels("contract_number"))   # from base
        self.assertEqual(perpres.get("part_hints", {}).get("ssuk"), "general_terms")  # its own

    def test_profile_overlay_extends_rather_than_replaces(self) -> None:
        base_labels = for_profile(None).labels("contract_number")
        overlaid = for_profile({"profile_id": "x", "vocabulary": {
            "label_dictionaries": {"contract_number": ["Nomor Akad"]},
        }})
        self.assertEqual(overlaid.labels("contract_number"), ["Nomor Akad"])
        self.assertEqual(overlaid.labels("contract_name"), for_profile(None).labels("contract_name"))
        self.assertIn("Nomor Kontrak", base_labels)  # the base itself is untouched

    def test_unknown_profile_id_falls_back_to_the_base_layer(self) -> None:
        self.assertEqual(for_profile_id("no_such_profile").data, for_profile(None).data)

    def test_merged_view_covers_every_profile(self) -> None:
        merged = for_all_profiles()
        for profile in known_profiles():
            for part in (profile.get("vocabulary", {}).get("part_hints") or {}):
                self.assertIn(part, merged.get("part_hints"))


class CrossLayerConsistencyTests(unittest.TestCase):
    """Every part and subtype a profile can produce must be nameable downstream."""

    def test_every_profile_part_has_a_display_label(self) -> None:
        merged = for_all_profiles()
        labels = merged.get("part_display_labels") or {}
        for profile in known_profiles():
            for marker in profile.get("sub_document_markers") or []:
                self.assertIn(
                    marker["name"], labels,
                    f"{profile['profile_id']} declares sub-document {marker['name']!r} with no display label; "
                    "retrieval/chat.py would show the raw key to a reader",
                )

    def test_every_part_hint_points_at_a_real_sub_document(self) -> None:
        declared = {m["name"] for p in known_profiles() for m in p.get("sub_document_markers") or []}
        for term, part in (for_all_profiles().get("part_hints") or {}).items():
            self.assertIn(part, declared, f"part hint {term!r} resolves to unknown sub-document {part!r}")

    def test_every_amount_subtype_has_a_display_label(self) -> None:
        merged = for_all_profiles()
        labels = merged.get("subtype_display_labels") or {}
        for key in ("duration_subtypes", "monetary_subtypes", "penalty_subtypes"):
            for entry in merged.get(key) or []:
                self.assertIn(
                    entry["subtype"], labels,
                    f"{key} entry {entry['subtype']!r} has no display label; retrieval/lookup.py could "
                    "extract it but never name it",
                )

    def test_clause_scope_names_a_declared_sub_document(self) -> None:
        for profile in known_profiles():
            scope = (profile.get("expected_invariants") or {}).get("clause_sequence_scope")
            if scope is None:
                continue
            names = {m["name"] for m in profile.get("sub_document_markers") or []}
            self.assertIn(scope, names, f"{profile['profile_id']}: clause_sequence_scope {scope!r} is not declared")

    def test_every_vocabulary_pattern_compiles(self) -> None:
        merged = for_all_profiles()
        for key in ("document_type_signals", "subtype_signals"):
            for entry in merged.get(key) or []:
                for pattern in entry["patterns"]:
                    re.compile(pattern)
        for key in ("party_role_markers", "date_context_labels", "duration_subtypes",
                    "monetary_subtypes", "penalty_subtypes"):
            for entry in merged.get(key) or []:
                re.compile(entry["pattern"])
        for pattern in merged.get("running_header_patterns") or []:
            re.compile(pattern)


class NoDomainWordsInStageCodeTests(unittest.TestCase):
    """Acceptance criterion 2 of the review: the stage code that decides tree
    shape holds no domain vocabulary. Those files should read the same whether
    the contract is Indonesian, and whatever family it belongs to."""

    STAGE_FILES = ("tree.py", "numbering.py", "layout.py", "blocks.py")
    # Family vocabulary: these name one contract template, never a language.
    FORBIDDEN = ("LAMPIRAN", "SSUK", "SSKK", "PIHAK", "PERJANJIAN")
    # Language vocabulary that is still in code, knowingly: `numbering.py`
    # matches the Indonesian heading words "PASAL", "BAB" and "BAGIAN" as part
    # of its numbering grammar. They belong in the locale layer, and moving them
    # is Phase 3 work (relative depth), so this test pins where they are rather
    # than pretending they are gone.
    KNOWN_LANGUAGE_WORDS_IN_CODE = {"numbering.py": {"PASAL", "BAB", "BAGIAN"}}

    def test_known_language_words_are_still_only_in_numbering(self) -> None:
        for name in self.STAGE_FILES:
            source = (PIPELINE_DIR / name).read_text(encoding="utf-8")
            code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
            found = {w for w in ("PASAL", "BAB", "BAGIAN") if re.search(rf"\b{w}\b", code)}
            self.assertEqual(
                found, self.KNOWN_LANGUAGE_WORDS_IN_CODE.get(name, set()),
                f"{name}: Indonesian heading words moved or spread — update Phase 3's scope",
            )

    def test_stage_files_carry_no_family_vocabulary(self) -> None:
        offenders = []
        for name in self.STAGE_FILES:
            source = (PIPELINE_DIR / name).read_text(encoding="utf-8")
            for line_no, line in enumerate(source.splitlines(), start=1):
                code = line.split("#", 1)[0]          # a comment may name an example
                for word in self.FORBIDDEN:
                    if word in code:
                        offenders.append(f"{name}:{line_no} {word}")
        self.assertEqual(offenders, [], "domain vocabulary belongs in profiles/, not in the stage code")


class LayerBoundaryTests(unittest.TestCase):
    def test_vocabulary_imports_neither_half(self) -> None:
        """It sits under both, so importing either would create the cycle the
        architecture forbids."""
        source = (REPO / "vocabulary" / "__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                self.assertFalse(name.startswith(("pipeline", "retrieval")),
                                 f"vocabulary must not import {name}")

    def test_retrieval_still_does_not_import_pipeline(self) -> None:
        offenders = []
        for path in sorted(RETRIEVAL_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                offenders += [f"{path.name}: {m}" for m in modules if m.startswith("pipeline")]
        self.assertEqual(offenders, [], "retrieval must reach shared vocabulary through `vocabulary`, not pipeline")


if __name__ == "__main__":
    unittest.main()
