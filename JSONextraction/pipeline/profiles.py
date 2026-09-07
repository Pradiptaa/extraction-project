"""Stage 3 — PROFILE SELECTION. Scores registered profiles against document
features; best score above threshold wins, else `generic_contract_v1` (which
always exists and always populates `core`, even with a shallow tree).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
FALLBACK_PROFILE_ID = "generic_contract_v1"


@dataclass
class ProfileMatch:
    profile: dict
    score: float


def load_profiles(profile_dir: Path = DEFAULT_PROFILE_DIR) -> list[dict]:
    profiles = []
    for path in sorted(profile_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            profiles.append(json.load(f))
    if not any(p["profile_id"] == FALLBACK_PROFILE_ID for p in profiles):
        raise RuntimeError(f"{FALLBACK_PROFILE_ID}.json is required and was not found in {profile_dir}")
    return profiles


def _score_profile(profile: dict, full_text: str, page_count: int, dominant_layout: str) -> float:
    score = 0.0
    for signal in profile.get("match_signals", []):
        weight = signal.get("weight", 0.0)
        stype = signal["type"]
        if stype == "text_contains":
            if re.search(re.escape(signal["value"]), full_text, re.IGNORECASE):
                score += weight
        elif stype == "layout":
            if signal["value"] == "two_column_dominant" and dominant_layout == "two_column":
                score += weight
            elif signal["value"] == dominant_layout:
                score += weight
        elif stype == "page_count_range":
            lo, hi = signal["value"]
            if lo <= page_count <= hi:
                score += weight
    return score


def select_profile(profiles: list[dict], full_text: str, page_count: int, dominant_layout: str) -> ProfileMatch:
    fallback = next(p for p in profiles if p["profile_id"] == FALLBACK_PROFILE_ID)
    best = ProfileMatch(fallback, 0.0)
    for profile in profiles:
        if profile["profile_id"] == FALLBACK_PROFILE_ID:
            continue
        score = _score_profile(profile, full_text, page_count, dominant_layout)
        if score >= profile.get("match_threshold", 0.6) and score > best.score:
            best = ProfileMatch(profile, score)
    return best
