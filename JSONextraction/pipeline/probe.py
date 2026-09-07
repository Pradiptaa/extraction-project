"""Stage 1 — PROBE. Per-page geometry, font presence, char density, image
coverage, ruling-line count, and a word x0 histogram. No text decisions here —
this only gathers signals that later stages route and classify on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pdfplumber


@dataclass
class PageProbe:
    page: int
    width: float
    height: float
    rotation: int
    char_count: int
    word_count: int
    image_count: int
    image_coverage: float          # fraction of page area covered by images
    fonts: list[str]
    ruling_line_count: int
    words: list[dict] = field(default_factory=list)  # [{text, x0, top, x1, bottom, size, fontname}]


def probe_document(pdf_path: str) -> list[PageProbe]:
    probes: list[PageProbe] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size"])
            chars = page.chars
            page_area = float(page.width) * float(page.height) or 1.0
            image_area = sum(
                max(0.0, (img.get("x1", 0) - img.get("x0", 0)))
                * max(0.0, (img.get("bottom", 0) - img.get("top", 0)))
                for img in page.images
            )
            fonts = sorted({c.get("fontname", "") for c in chars if c.get("fontname")})
            ruling_lines = len(page.lines) + len(page.rects)

            probes.append(
                PageProbe(
                    page=i,
                    width=float(page.width),
                    height=float(page.height),
                    rotation=int(page.rotation or 0),
                    char_count=len(chars),
                    word_count=len(words),
                    image_count=len(page.images),
                    image_coverage=min(1.0, image_area / page_area),
                    fonts=fonts,
                    ruling_line_count=ruling_lines,
                    words=[
                        {
                            "text": w["text"],
                            "x0": w["x0"],
                            "top": w["top"],
                            "x1": w["x1"],
                            "bottom": w["bottom"],
                            "size": w.get("size", 0.0),
                            "fontname": w.get("fontname", ""),
                        }
                        for w in words
                    ],
                )
            )
    return probes
