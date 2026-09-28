"""Domain vocabulary, loaded from `profiles/` and shared by both halves. """
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
BASE_VOCABULARY_ID = "base_id"

_MERGED_MAPPINGS = (
    "label_dictionaries", "subtype_display_labels", "field_display_labels",
    "part_hints", "part_display_labels",
)
_EXTENDED_LISTS = (
    "document_type_signals", "subtype_signals", "party_role_markers", "self_reference_terms",
    "date_context_labels", "duration_subtypes", "monetary_subtypes", "penalty_subtypes",
    "reference_number_labels", "running_header_patterns",
)


@dataclass(frozen=True)
class Vocabulary:

    profile_id: str | None
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def labels(self, field_name: str) -> list[str]:
        return list((self.data.get("label_dictionaries") or {}).get(field_name) or [])

    def patterns(self, key: str, flags: int = re.IGNORECASE) -> list[tuple[re.Pattern, dict]]:
        out = []
        for entry in self.data.get(key) or []:
            pattern = entry.get("pattern") if isinstance(entry, dict) else entry
            if pattern:
                out.append((re.compile(pattern, flags), entry if isinstance(entry, dict) else {}))
        return out

    def keyword_table(self, key: str, label_field: str = "subtype") -> list[tuple[re.Pattern, str]]:
        return [(pattern, entry.get(label_field, "")) for pattern, entry in self.patterns(key)]

    def display_label(self, subtype: str | None, field_name: str | None = None) -> str:
        labels = self.data.get("subtype_display_labels") or {}
        if subtype and subtype in labels:
            return labels[subtype]
        fields = self.data.get("field_display_labels") or {}
        return fields.get(field_name or "", field_name or "")


def _merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key.startswith("_"):
            continue
        if key in _MERGED_MAPPINGS and isinstance(value, dict):
            merged[key] = {**(base.get(key) or {}), **value}
        elif key in _EXTENDED_LISTS and isinstance(value, list):
            existing = list(base.get(key) or [])
            merged[key] = existing + [item for item in value if item not in existing]
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_base(profile_dir: str | None = None) -> dict:
    path = Path(profile_dir or PROFILE_DIR) / f"{BASE_VOCABULARY_ID}.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def for_profile(profile: dict | None, profile_dir: str | None = None) -> Vocabulary:
    base = load_base(profile_dir)
    if not profile:
        return Vocabulary(None, dict(base))
    overlay = profile.get("vocabulary") or {}
    return Vocabulary(profile.get("profile_id"), _merge(base, overlay))


def for_profile_id(profile_id: str | None, profile_dir: str | None = None) -> Vocabulary:
    if not profile_id:
        return for_profile(None, profile_dir)
    directory = Path(profile_dir or PROFILE_DIR)
    for path in sorted(directory.glob("*.json")):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate.get("profile_id") == profile_id:
            return for_profile(candidate, profile_dir)
    return for_profile(None, profile_dir)


def for_all_profiles(profile_dir: str | None = None) -> Vocabulary:
    data = load_base(profile_dir)
    for profile in known_profiles(profile_dir):
        data = _merge(data, profile.get("vocabulary") or {})
    return Vocabulary(None, data)


def known_profiles(profile_dir: str | None = None) -> list[dict]:
    directory = Path(profile_dir or PROFILE_DIR)
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate.get("profile_id"):
            out.append(candidate)
    return out
