"""Stage 2 — ROUTE. Decided per page, never per document (analisis_pipeline_kontrak.md A.2).

v1 scope: this codebase does not ship an LLM fallback, and OCR is treated the
same way — no OCR engine is wired in. Pages that would need OCR are extracted
with whatever native text exists (usually none) and flagged
`ocr_required_not_available` so the gap is visible in the quality report
instead of silently producing an empty node.
"""
from __future__ import annotations

from dataclasses import dataclass

from .probe import PageProbe

MIN_CHARS_FOR_NATIVE = 50
MIN_IMAGE_COVERAGE_FOR_OCR = 0.80
MAX_CHARS_FOR_OCR_CANDIDATE = 5


@dataclass
class RouteDecision:
    page: int
    method: str            # native | ocr_needed | hybrid_review
    reason: str


def route_pages(probes: list[PageProbe]) -> list[RouteDecision]:
    decisions = []
    for p in probes:
        if p.char_count > MIN_CHARS_FOR_NATIVE and p.fonts:
            decisions.append(RouteDecision(p.page, "native", "sufficient text and fonts present"))
        elif p.char_count <= MAX_CHARS_FOR_OCR_CANDIDATE and p.image_coverage >= MIN_IMAGE_COVERAGE_FOR_OCR:
            decisions.append(
                RouteDecision(
                    p.page,
                    "ocr_needed",
                    f"char_count={p.char_count} image_coverage={p.image_coverage:.2f} "
                    "and no OCR engine wired in v1",
                )
            )
        else:
            decisions.append(
                RouteDecision(
                    p.page,
                    "hybrid_review",
                    f"ambiguous signals: char_count={p.char_count} fonts={len(p.fonts)} "
                    f"image_coverage={p.image_coverage:.2f}",
                )
            )
    return decisions
