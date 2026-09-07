"""Shared value-object helpers and ID generation for the extraction schema.

Every field promoted into `core` (and most entities) follows the same
value-object shape: {value, raw, confidence, method, evidence, candidates, flags}.
Centralizing construction here keeps that contract from drifting between modules.
"""
from __future__ import annotations

from itertools import count
from typing import Any, Optional

SCHEMA_VERSION = "1.0.0"


def value_object(
    value: Any = None,
    raw: Optional[str] = None,
    confidence: float = 0.0,
    method: str = "unresolved",
    evidence: Optional[dict] = None,
    candidates: Optional[list] = None,
    flags: Optional[list] = None,
    **extra: Any,
) -> dict:
    obj = {
        "value": value,
        "raw": raw,
        "confidence": round(confidence, 4),
        "method": method,
        "evidence": evidence,
        "candidates": candidates or [],
        "flags": flags or [],
    }
    obj.update(extra)
    return obj


class NodeIdGenerator:
    """Sequential, opaque node IDs — n_0001, n_0002, ... No domain vocabulary."""

    def __init__(self) -> None:
        self._counter = count(1)

    def next(self) -> str:
        return f"n_{next(self._counter):04d}"


class ReadingOrderCounter:
    def __init__(self) -> None:
        self._counter = count(1)

    def next(self) -> int:
        return next(self._counter)
