"""OCR pipeline — a parallel front end to `main.py`, for scanned/image-only PDFs.

Design (see the OCR design discussion): the existing pipeline's only
PDF-specific dependencies are at the *front* — `probe.probe_document()`, which
builds `PageProbe` objects, and `blocks.extract_table_blocks()`, which needs
pdfplumber's vector lines. Everything from `layout.classify_layout()` onward
consumes `PageProbe` and knows nothing about where the words came from.

So this module reimplements exactly those two front-end pieces on top of
Tesseract + OpenCV, then calls the SAME downstream stages `main.py` calls. That
is what makes the output format identical by construction rather than by
convention — and it means a fix to `tree.py` benefits both pipelines instead of
having to be applied twice.

Nothing in the existing pipeline is modified. This module only imports.

    python -m pipeline.ocr_main "scan.pdf" --out output_ocr

Coordinate contract (the critical detail): Tesseract reports pixel boxes at
render DPI, but every geometry heuristic downstream is calibrated in PDF points
— the 6pt row bucket in `blocks.py` and `right_column_start_frac` in
`layout.py`. All OCR boxes are therefore scaled back to PDF points before a
`PageProbe` is constructed, so those thresholds keep the meaning they were
tuned for.

Known differences from the native pipeline, all deliberate and flagged in the
output rather than hidden:
  - No font information exists. `fontname` is "ocr" for every word, so
    `blocks._line_to_block`'s bold detection always yields False and
    `is_bold` is dead weight on this path.
  - `font_size` is derived from the OCR box height, which tracks glyph height
    rather than point size. It is proportional but not equal to the native value.
  - Ruled tables are found by morphological line detection, not pdfplumber cell
    reconstruction (`extraction_method: opencv_ruled_ocr`).
  - `dual_parser_oracle` is not applicable — see `_neutralize_oracle_check`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyMuPDF is required for the OCR pipeline: pip install -r requirements.txt"
    ) from exc

try:
    import pytesseract
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pytesseract is required for the OCR pipeline: pip install -r requirements.txt"
    ) from exc

from . import core_fields, entities as entities_mod, profiles as profiles_mod
from .blocks import TableBlock, extract_text_blocks
from .layout import classify_layout
from .main import (
    assign_sub_documents,
    build_table_entries,
    extract_page_label,
    guess_document_status,
    prelim_page_text,
    sha256_of,
)
from .probe import PageProbe
from .schema import SCHEMA_VERSION
from .tree import build_tree
from .validate import run_validation

DEFAULT_DPI = 300
DEFAULT_LANG = "ind+eng"
DEFAULT_MIN_CONF = 30.0  # Tesseract 0-100 per-word confidence floor

# Two-pass recognition. `--psm 3` (Tesseract's own automatic page segmentation)
# cannot be used: it treats the narrow subclause-label gutter as a separate
# region and discards it wherever the left column is empty beside it, deleting
# labels like "21.4" entirely — not at low confidence, simply absent. Without
# the label `numbering.py` creates no node and the text is absorbed into its
# parent, which is what collapsed 738 nodes to 569 on the first run.
#
# No single mode is safe, though. Measured across three two-column pages:
#   psm 3      loses 21.4-21.7 on p19
#   psm 4 / 6  recover those but lose 24.4 on p21
#   psm 11/12  recover every decimal label, but degrade elsewhere — commas read
#              as semicolons, adjacent words merge, and "a." / "b." lose the
#              trailing period that `numbering.py` needs for `letter_dotted`
#
# So pass 1 (PSM 4) supplies word segmentation and punctuation, and pass 2
# (PSM 11, sparse text) contributes ONLY tokens whose boxes pass 1 missed
# entirely. Where both saw a word, pass 1's reading wins. This recovers the
# dropped gutter labels without importing pass 2's transcription damage, and
# assumes nothing about where on the page labels sit.
DEFAULT_PSM = 4
DEFAULT_SECONDARY_PSM = 11
# A pass-2 box is treated as already-found when it overlaps a pass-1 box by
# this fraction of the smaller of the two areas.
MERGE_OVERLAP_RATIO = 0.30

# Deskew search. The estimate comes from a projection profile — the page is
# rotated through candidate angles and scored on the variance of its horizontal
# ink projection, which peaks when text baselines are level. This measures
# baselines directly, unlike `cv2.minAreaRect` over the ink cloud, which
# measures the bounding box of the ink and returns confident nonsense on any
# asymmetric page (a table, a signature block, ragged margins). That estimator
# rotated 7 of 74 pages of a geometrically perfect born-digital render, one of
# them by -1.6 degrees, costing that page 60% of its text.
MIN_DESKEW_DEG = 0.3     # below this the correction is noise; leave the page alone
MAX_DESKEW_DEG = 3.0
DESKEW_STEP_DEG = 0.1
DESKEW_SCORE_WIDTH = 800  # downsample before the angle search; it is a shape measure

# Morphological line detection: the opening kernel is this fraction of the page,
# which is what separates rules from glyph strokes.
LINE_LEN_FRACTION = 25
LINE_CLUSTER_TOL_PT = 3.0

# Grid qualification. Morphological opening alone cannot tell a table rule from
# the strokes of a letterhead emblem: on pages 1, 5 and 6 the government crest
# produces 9-21 "rules" of 4-8% of the page, which a raw count (or a count
# inflated by implied cell area) reads as a dense table. Those pages then route
# their body text into tables[] and lose it — p6 lost 876 of 1999 characters.
#
# Measured separation is wide and unambiguous. On every genuine table page the
# horizontal rules span 67-71% of the page width and the vertical rules run the
# table's full height; the emblem strokes span 4-8% and connect nothing.
#
# A rule therefore has to earn its place structurally: a horizontal rule counts
# only if it is long, and a vertical rule counts only if it CROSSES at least two
# long horizontal rules — which is what makes a closed row of cells, and is the
# thing an emblem, an underline or a signature line can never do.
MIN_RULE_SPAN_FRAC = 0.20   # of page width, for a horizontal rule to count
# Collinear segments join into one rule only across a gap this small. Unioning
# without it invents rules: on page 1 four separate strokes near x=115-122 (part
# of the crest at the top, part of the footer 750pt below) collapsed into a
# single full-height "rule" that appeared to cross both the header and footer
# underlines. Dashes and scan breaks are a few points wide; 750 is not a dash.
RULE_JOIN_GAP_PT = 12.0
MIN_CROSSED_RULES = 2       # long h-rules a v-rule must cross to count
MIN_GRID_RULES = 2          # qualifying rules needed on each axis
# A 2x2 set of rules encloses exactly one cell — a bordered box, not a table.
# Page 1's footer box is precisely that and must stay `form`; the smallest
# genuine tables here are one row of two or three columns (pages 69 and 66).
MIN_GRID_CELLS = 2
RULE_CROSS_TOL_PT = 2.0
# `layout.classify_layout` routes to `ruled_table` at ruling_line_count >= 20.
# Once a grid is confirmed structurally, report a count that clears that
# threshold; pages without one report only what was actually found.
RULED_TABLE_SIGNAL = 20
# A gap this many times the median row gap is read as the boundary between two
# stacked tables rather than an unusually tall row.
TABLE_SPLIT_GAP_RATIO = 3.0


# --------------------------------------------------------------------------
# Stage 1 (OCR) — render, preprocess, recognize, and rebuild PageProbe
# --------------------------------------------------------------------------

def _render_gray(page, dpi: int) -> np.ndarray:
    """Render a page to a grayscale numpy image at the requested DPI."""
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return np.ascontiguousarray(img)


def _rotate(img: np.ndarray, angle: float, border: int) -> np.ndarray:
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def _estimate_skew(gray: np.ndarray) -> float:
    """Projection-profile skew estimate. Text lines project into sharp peaks and
    troughs only when they are level, so the horizontal projection's variance is
    maximal at the true skew angle."""
    scale = min(1.0, DESKEW_SCORE_WIDTH / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(
        cv2.bitwise_not(small), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    if not mask.any():
        return 0.0

    best_angle, best_score = 0.0, -1.0
    steps = int(round(2 * MAX_DESKEW_DEG / DESKEW_STEP_DEG)) + 1
    for i in range(steps):
        angle = -MAX_DESKEW_DEG + i * DESKEW_STEP_DEG
        # Rotating the mask on a black border keeps the projection measuring
        # ink only — a replicated border would smear edge rows into the profile.
        projection = _rotate(mask, angle, border=0).sum(axis=1, dtype=np.float64)
        score = float(projection.var())
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the page so text baselines are horizontal. Returns the corrected
    image and the angle applied (0.0 when the estimate is below the noise floor,
    which is the expected outcome for any born-digital render)."""
    angle = _estimate_skew(gray)
    if abs(angle) < MIN_DESKEW_DEG:
        return gray, 0.0
    return _rotate(gray, angle, border=255), angle


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold after a light blur — stable across scan qualities without
    the block-size tuning adaptive thresholding needs."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _cluster_segments(
    segments: list[tuple[float, float, float]], tolerance: float
) -> list[tuple[float, float, float]]:
    """Collapse near-collinear segments into one rule each, unioning their
    extents so a rule broken into pieces (dashed, or split by a scan artefact)
    is measured at its true length."""
    if not segments:
        return []
    ordered = sorted(segments)
    groups: list[list[tuple[float, float, float]]] = [[ordered[0]]]
    for seg in ordered[1:]:
        if seg[0] - groups[-1][-1][0] <= tolerance:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    rules: list[tuple[float, float, float]] = []
    for group in groups:
        position = sum(s[0] for s in group) / len(group)
        # Within one collinear group, join only segments that are actually
        # contiguous; a large gap means two unrelated marks that happen to share
        # an axis, not one broken rule.
        by_extent = sorted(group, key=lambda s: s[1])
        start, end = by_extent[0][1], by_extent[0][2]
        for seg in by_extent[1:]:
            if seg[1] - end <= RULE_JOIN_GAP_PT:
                end = max(end, seg[2])
            else:
                rules.append((position, start, end))
                start, end = seg[1], seg[2]
        rules.append((position, start, end))
    return rules


def _detect_rule_segments(
    binary: np.ndarray, scale_x: float, scale_y: float
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Find candidate rules via morphological opening, as (position, start, end)
    triples in PDF points — (y, x0, x1) horizontally and (x, y0, y1) vertically.
    Keeping the extents is the point: discarding them is what made an emblem
    stroke indistinguishable from a table rule."""
    ink = cv2.bitwise_not(binary)  # rules are dark on light; work on the inverse
    h, w = ink.shape

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // LINE_LEN_FRACTION), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // LINE_LEN_FRACTION)))
    h_mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel, iterations=1)

    h_segments, v_segments = [], []
    for contour in cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(contour)
        h_segments.append(((y + ch / 2.0) * scale_y, x * scale_x, (x + cw) * scale_x))
    for contour in cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(contour)
        v_segments.append(((x + cw / 2.0) * scale_x, y * scale_y, (y + ch) * scale_y))

    return (
        _cluster_segments(h_segments, LINE_CLUSTER_TOL_PT),
        _cluster_segments(v_segments, LINE_CLUSTER_TOL_PT),
    )


def _crosses(v: tuple[float, float, float], h: tuple[float, float, float]) -> bool:
    """True when vertical rule `v` and horizontal rule `h` actually meet."""
    x, y0, y1 = v
    y, hx0, hx1 = h
    return (
        hx0 - RULE_CROSS_TOL_PT <= x <= hx1 + RULE_CROSS_TOL_PT
        and y0 - RULE_CROSS_TOL_PT <= y <= y1 + RULE_CROSS_TOL_PT
    )


def _detect_rule_grid(
    binary: np.ndarray, scale_x: float, scale_y: float, page_width: float
) -> tuple[list[float], list[float], bool]:
    """Qualify candidate rules into an actual cell grid.

    Returns the surviving horizontal and vertical rule positions, plus whether
    they constitute a grid. See MIN_RULE_SPAN_FRAC above for why length alone is
    not enough and crossing is the real test.
    """
    h_segments, v_segments = _detect_rule_segments(binary, scale_x, scale_y)

    long_h = [s for s in h_segments if (s[2] - s[1]) >= MIN_RULE_SPAN_FRAC * page_width]
    good_v = [v for v in v_segments if sum(_crosses(v, h) for h in long_h) >= MIN_CROSSED_RULES]
    # Re-qualify horizontals symmetrically: a lone long rule that nothing
    # crosses is a header underline, not part of a grid.
    good_h = [h for h in long_h if sum(_crosses(v, h) for v in good_v) >= MIN_GRID_RULES]

    cells = max(0, len(good_h) - 1) * max(0, len(good_v) - 1)
    if len(good_h) >= MIN_GRID_RULES and len(good_v) >= MIN_GRID_RULES and cells >= MIN_GRID_CELLS:
        return sorted(h[0] for h in good_h), sorted(v[0] for v in good_v), True

    # Report what was genuinely found so `has_ruling_lines` stays truthful, but
    # nothing that would trip the ruled_table threshold.
    return sorted(h[0] for h in long_h), [], False


def _ocr_words(
    binary: np.ndarray, scale_x: float, scale_y: float, lang: str, psm: int, min_conf: float
) -> tuple[list[dict], list[float]]:
    """Run Tesseract and convert its pixel word boxes into the point-space word
    dicts `PageProbe.words` is specified in."""
    data = pytesseract.image_to_data(
        binary, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )

    words: list[dict] = []
    confidences: list[float] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue

        left, top = float(data["left"][i]), float(data["top"][i])
        width, height = float(data["width"][i]), float(data["height"][i])
        words.append(
            {
                "text": text,
                "x0": left * scale_x,
                "top": top * scale_y,
                "x1": (left + width) * scale_x,
                "bottom": (top + height) * scale_y,
                # Glyph-box height in points: proportional to, but not equal to,
                # the native `size` attribute pdfplumber reports.
                "size": height * scale_y,
                "fontname": "ocr",
            }
        )
        confidences.append(conf)

    return words, confidences


def _find_overlap(word: dict, others: list[dict]) -> int | None:
    """Index of this box's counterpart among `others`, or None. Overlap is
    measured against the smaller of the two areas so a tight label box sitting
    inside a looser one still counts as the same token."""
    area = max(1e-6, (word["x1"] - word["x0"]) * (word["bottom"] - word["top"]))
    for i, other in enumerate(others):
        ix = min(word["x1"], other["x1"]) - max(word["x0"], other["x0"])
        if ix <= 0:
            continue
        iy = min(word["bottom"], other["bottom"]) - max(word["top"], other["top"])
        if iy <= 0:
            continue
        other_area = max(1e-6, (other["x1"] - other["x0"]) * (other["bottom"] - other["top"]))
        if (ix * iy) / min(area, other_area) >= MERGE_OVERLAP_RATIO:
            return i
    return None


_TRAILING_PUNCT = ".,;:)]}"


def _skeleton(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _reconcile(text1: str, conf1: float, text2: str, conf2: float) -> tuple[str, float]:
    """Choose between two passes' readings of the same box.

    Confidence alone is not a safe tie-break. PSM 11 is systematically both MORE
    confident and worse at punctuation, so a naive "higher confidence wins" rule
    replaces correct readings with damaged ones — it turned the clause label
    "3." (PSM 4, conf 92) into "3" (PSM 11, conf 96), and `numbering.py` cannot
    recognize a clause without the period. That single substitution cost six
    clause nodes.

    Two guards, in order:
      1. The passes must agree on the token's alphanumeric content. If they do
         not, this is a segmentation or substitution disagreement ("Built dan"
         vs "Builfdan") and the primary pass — which has the better word
         segmentation — is kept unconditionally.
      2. If they differ ONLY in trailing punctuation, keep the richer reading
         ("3." over "3", "a." over "a"). Ties go to the primary pass, which
         also keeps its comma when pass 2 offers a semicolon.

    Only when punctuation differs *internally* — "214" vs "21.4", where the
    period is load-bearing rather than terminal — does confidence decide.
    """
    if text1 == text2:
        return text1, conf1
    if _skeleton(text1) != _skeleton(text2):
        return text1, conf1
    if text1.rstrip(_TRAILING_PUNCT) == text2.rstrip(_TRAILING_PUNCT):
        return (text1, conf1) if len(text1) >= len(text2) else (text2, conf2)
    return (text2, conf2) if conf2 > conf1 else (text1, conf1)


def _normalize_row_tops(words: list[dict]) -> None:
    """Give every word on a visual row the same `top`/`bottom`, in place.

    This is a geometry-schema conversion, not a heuristic. pdfplumber reports a
    word's LINE BOX top, so every word in a row shares one `top` to the
    hundredth of a point. Tesseract reports the INK top, which moves 2-3pt
    within a single row depending on whether a word carries capitals or
    ascenders ("dapat" 105.36, "214" 105.84, "RMPK" 106.08 on page 19).

    Downstream code is written against the pdfplumber convention and breaks
    without it: `layout._line_groups` sorts by `(top, x0)` and `layout.py` then
    reads `line[0]` as the row's LEADING word, which is only true when the tops
    are equal. With ink tops, `line[0]` is whichever word has the tallest
    letters, and rows split mid-line — which erased the entire subclause-label
    column (histogram bin 18) from page 19 and left `right_column_start_frac`
    pointing at the body indent instead of the labels.

    Rows are clustered on vertical centre, which is far more stable than either
    edge, with a tolerance derived from the page's own median word height so it
    scales with font size rather than assuming one.
    """
    if not words:
        return

    heights = sorted(w["bottom"] - w["top"] for w in words)
    median_height = heights[len(heights) // 2] or 1.0
    tolerance = 0.4 * median_height

    ordered = sorted(words, key=lambda w: (w["top"] + w["bottom"]) / 2.0)
    row: list[dict] = [ordered[0]]
    rows: list[list[dict]] = [row]
    row_centre = (ordered[0]["top"] + ordered[0]["bottom"]) / 2.0

    for w in ordered[1:]:
        centre = (w["top"] + w["bottom"]) / 2.0
        if abs(centre - row_centre) <= tolerance:
            row.append(w)
            # Track the running mean so a row does not drift on a chain of
            # individually-small steps the way last-word comparison would.
            row_centre = sum((x["top"] + x["bottom"]) / 2.0 for x in row) / len(row)
        else:
            row = [w]
            rows.append(row)
            row_centre = centre

    for row in rows:
        top = min(w["top"] for w in row)
        bottom = max(w["bottom"] for w in row)
        for w in row:
            w["top"] = top
            w["bottom"] = bottom


def _ocr_words_two_pass(
    binary: np.ndarray,
    scale_x: float,
    scale_y: float,
    lang: str,
    psm: int,
    secondary_psm: int | None,
    min_conf: float,
) -> tuple[list[dict], list[float], int, int]:
    """Primary recognition pass plus a sparse-text recovery pass.

    Two ways the second pass contributes:
      - **recovered**: a box the primary pass missed entirely, appended as-is.
      - **corrected**: both passes found the same box but disagree on the text,
        and the second pass is more confident. This matters more than the
        recovery case in practice — on page 19 PSM 4 reads the subclause label
        "21.4" as "214", dropping the decimal point, which `numbering.py` then
        cannot recognize as a label at all. PSM 11 reads it correctly at higher
        confidence (89 vs 86). The tie-break is confidence alone, so nothing
        here needs to know what a label looks like.

    Returns (words, confidences, recovered, corrected).
    """
    words, confidences = _ocr_words(binary, scale_x, scale_y, lang, psm, min_conf)
    if secondary_psm is None or secondary_psm == psm:
        return words, confidences, 0, 0

    extra_words, extra_confs = _ocr_words(binary, scale_x, scale_y, lang, secondary_psm, min_conf)
    recovered = corrected = 0
    for word, conf in zip(extra_words, extra_confs):
        index = _find_overlap(word, words)
        if index is None:
            words.append(dict(word, fontname="ocr_pass2"))
            confidences.append(conf)
            recovered += 1
        else:
            chosen, chosen_conf = _reconcile(
                words[index]["text"], confidences[index], word["text"], conf
            )
            if chosen != words[index]["text"]:
                # Keep the primary pass's box (both passes agree on it by
                # construction); take only the better transcription.
                words[index] = dict(words[index], text=chosen, fontname="ocr_pass2_text")
                confidences[index] = chosen_conf
                corrected += 1

    return words, confidences, recovered, corrected


def probe_document_ocr(
    pdf_path: str,
    dpi: int = DEFAULT_DPI,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    secondary_psm: int | None = DEFAULT_SECONDARY_PSM,
    min_conf: float = DEFAULT_MIN_CONF,
    deskew: bool = True,
    debug_dir: Path | None = None,
    progress: bool = True,
) -> tuple[list[PageProbe], dict[int, dict], dict[int, dict]]:
    """OCR analogue of `probe.probe_document()`.

    Returns (probes, rule_lines_by_page, ocr_stats_by_page). The rule-line
    positions are handed on to table extraction so OpenCV runs once per page,
    not twice.
    """
    probes: list[PageProbe] = []
    rules_by_page: dict[int, dict] = {}
    stats_by_page: dict[int, dict] = {}

    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            if progress:
                print(f"  OCR page {index}/{doc.page_count}...", file=sys.stderr, flush=True)

            page_w, page_h = float(page.rect.width), float(page.rect.height)
            gray = _render_gray(page, dpi)
            # Derive the scale from the actual rendered size rather than
            # assuming dpi/72 exactly — rounding in the rasterizer would
            # otherwise put a sub-point systematic error into every box.
            scale_x = page_w / gray.shape[1]
            scale_y = page_h / gray.shape[0]

            skew_angle = 0.0
            if deskew:
                gray, skew_angle = _deskew(gray)
            binary = _binarize(gray)

            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / f"page_{index:03d}.png"), binary)

            words, confidences, recovered, corrected = _ocr_words_two_pass(
                binary, scale_x, scale_y, lang, psm, secondary_psm, min_conf
            )
            # Must run after the merge, so pass-2 boxes join the same rows.
            _normalize_row_tops(words)
            h_lines, v_lines, is_grid = _detect_rule_grid(binary, scale_x, scale_y, page_w)

            # `layout.classify_layout` routes a page to `ruled_table` at
            # ruling_line_count >= 20, a threshold calibrated against
            # pdfplumber's lines + rects (every table cell contributes a rect).
            # There is no honest way to reproduce that count from an image, so
            # the grid decision is made here — structurally, by `_detect_rule_grid`
            # — and the count is reported as a signal of that decision rather
            # than as a measurement pretending to be comparable.
            ruling_line_count = (
                RULED_TABLE_SIGNAL + len(h_lines) + len(v_lines) if is_grid
                # Capped below the threshold: a page with many unconnected long
                # rules must not back into `ruled_table` by count alone.
                else min(RULED_TABLE_SIGNAL - 1, len(h_lines) + len(v_lines))
            )

            char_count = sum(len(w["text"]) for w in words)
            image_count = len(page.get_images(full=True))

            probes.append(
                PageProbe(
                    page=index,
                    width=page_w,
                    height=page_h,
                    rotation=int(page.rotation or 0),
                    char_count=char_count,
                    word_count=len(words),
                    image_count=image_count,
                    # A rendered scan is full-page imagery by definition; the
                    # native path's coverage ratio has no analogue here and is
                    # informational only (nothing downstream reads it).
                    image_coverage=1.0 if image_count else 0.0,
                    fonts=[],
                    ruling_line_count=ruling_line_count,
                    words=words,
                )
            )
            rules_by_page[index] = {"h": h_lines, "v": v_lines}
            stats_by_page[index] = {
                "mean_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
                "min_confidence": round(min(confidences), 2) if confidences else 0.0,
                "words_kept": len(words),
                "words_recovered_pass2": recovered,
                "words_corrected_pass2": corrected,
                "deskew_deg": round(skew_angle, 3),
            }

    return probes, rules_by_page, stats_by_page


# --------------------------------------------------------------------------
# Stage 5 (OCR) — ruled tables from detected rules instead of vector lines
# --------------------------------------------------------------------------

def _split_row_groups(h_lines: list[float]) -> list[list[float]]:
    """Split a page's horizontal rules into per-table groups. Two stacked
    tables separated by prose show up as an outsized gap between rules; an
    unusually tall single row does not reach the same multiple of the median."""
    if len(h_lines) < 3:
        return [h_lines] if len(h_lines) >= 2 else []

    gaps = [b - a for a, b in zip(h_lines, h_lines[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 0:
        return [h_lines]

    groups: list[list[float]] = [[h_lines[0]]]
    for gap, line in zip(gaps, h_lines[1:]):
        if gap > TABLE_SPLIT_GAP_RATIO * median_gap:
            groups.append([line])
        else:
            groups[-1].append(line)
    return [g for g in groups if len(g) >= 2]


def extract_table_blocks_ocr(
    probes: list[PageProbe], rules_by_page: dict[int, dict], page_numbers: list[int]
) -> dict[int, list[TableBlock]]:
    """OCR analogue of `blocks.extract_table_blocks()`. Cell text comes from the
    OCR words already recognized for the page — words are assigned to the cell
    their centre point falls inside, so a word straddling a rule lands in
    exactly one cell."""
    result: dict[int, list[TableBlock]] = {}
    if not page_numbers:
        return result

    probe_by_page = {p.page: p for p in probes}
    wanted = set(page_numbers)

    for page_no in sorted(wanted):
        probe = probe_by_page.get(page_no)
        rules = rules_by_page.get(page_no)
        if probe is None or rules is None:
            continue

        v_lines = rules["v"]
        if len(v_lines) < 2:
            continue

        page_tables: list[TableBlock] = []
        for row_lines in _split_row_groups(rules["h"]):
            x0, x1 = v_lines[0], v_lines[-1]
            top, bottom = row_lines[0], row_lines[-1]

            in_table = [
                w for w in probe.words
                if top <= (w["top"] + w["bottom"]) / 2.0 <= bottom
                and x0 <= (w["x0"] + w["x1"]) / 2.0 <= x1
            ]

            rows: list[list[str]] = []
            for r_top, r_bottom in zip(row_lines, row_lines[1:]):
                cells: list[str] = []
                for c_x0, c_x1 in zip(v_lines, v_lines[1:]):
                    cell_words = [
                        w for w in in_table
                        if r_top <= (w["top"] + w["bottom"]) / 2.0 <= r_bottom
                        and c_x0 <= (w["x0"] + w["x1"]) / 2.0 <= c_x1
                    ]
                    cell_words.sort(key=lambda w: (round(w["top"] / 6.0), w["x0"]))
                    cells.append(" ".join(w["text"] for w in cell_words).strip())
                rows.append(cells)

            if any(any(c for c in row) for row in rows):
                page_tables.append(
                    TableBlock(
                        page=page_no,
                        rows=rows,
                        bbox={"x0": x0, "top": top, "x1": x1, "bottom": bottom},
                        extraction_method="opencv_ruled_ocr",
                    )
                )

        if page_tables:
            result[page_no] = page_tables

    return result


# --------------------------------------------------------------------------
# Stage 9 adjustment
# --------------------------------------------------------------------------

def _neutralize_oracle_check(quality: dict) -> dict:
    """`validate.check_dual_parser_oracle` cross-checks against Poppler
    `pdftotext`, which on a scanned PDF returns an empty text layer — the
    comparison would report ~0 similarity on every page and raise a warning
    that says nothing about extraction quality. It only self-skips when the
    binary is absent, so the result is rewritten here instead. Done as a
    post-hoc edit of the returned dict specifically to keep `validate.py`
    untouched; a one-line optional kwarg there would be cleaner if edits to the
    shared pipeline ever become acceptable.
    """
    for check in quality["checks"]:
        if check["check"] == "dual_parser_oracle":
            check["status"] = "skip"
            check["detail"] = "not applicable: OCR source has no independent native text layer to compare against"
    quality["warn_count"] = sum(1 for c in quality["checks"] if c["status"] == "warn")
    return quality


# --------------------------------------------------------------------------
# Orchestration — mirrors main.run_pipeline stage for stage
# --------------------------------------------------------------------------

def run_ocr_pipeline(
    pdf_path: Path,
    output_dir: Path,
    profile_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    secondary_psm: int | None = DEFAULT_SECONDARY_PSM,
    min_conf: float = DEFAULT_MIN_CONF,
    deskew: bool = True,
    debug_dir: Path | None = None,
) -> dict:
    probes, rules_by_page, ocr_stats = probe_document_ocr(
        str(pdf_path), dpi=dpi, lang=lang, psm=psm, secondary_psm=secondary_psm,
        min_conf=min_conf, deskew=deskew, debug_dir=debug_dir,
    )

    layouts = {p.page: classify_layout(p) for p in probes}
    layout_type_by_page = {page: info.layout_type for page, info in layouts.items()}

    ruled_pages = [p for p, lt in layout_type_by_page.items() if lt == "ruled_table"]
    table_blocks_by_page = extract_table_blocks_ocr(probes, rules_by_page, ruled_pages)

    pages_blocks = {}
    for probe in probes:
        layout_type = layout_type_by_page[probe.page]
        if layout_type == "blank":
            pages_blocks[probe.page] = []
            continue
        if layout_type == "ruled_table":
            # Same rule as the native path: table *cell* text belongs in
            # tables[], but a heading or caption outside every table bbox is a
            # real node and also feeds sub-document marker detection.
            all_blocks = extract_text_blocks(probe, layouts[probe.page])
            table_bboxes = [t.bbox for t in table_blocks_by_page.get(probe.page, [])]
            pages_blocks[probe.page] = [
                b for b in all_blocks
                if not any(bbox["top"] - 2 <= b.top and b.bottom <= bbox["bottom"] + 2 for bbox in table_bboxes)
            ]
            continue
        pages_blocks[probe.page] = extract_text_blocks(probe, layouts[probe.page])

    page_order = sorted(p.page for p in probes)

    # Same reordering as main.run_pipeline: profile selection and
    # sub-document assignment must happen before build_tree so tree.py can
    # tell a genuine SSUK "clause" apart from an ordinary numbered list item
    # by which sub-document the page belongs to, not by page geometry.
    prelim_text_by_page = {page: prelim_page_text(pages_blocks.get(page, [])) for page in page_order}
    prelim_full_text = "\n\n".join(prelim_text_by_page.get(p, "") for p in page_order)

    layout_counts = Counter(lt for lt in layout_type_by_page.values() if lt != "blank")
    dominant_layout = layout_counts.most_common(1)[0][0] if layout_counts else "single_column"

    profile_list = profiles_mod.load_profiles(profile_dir or profiles_mod.DEFAULT_PROFILE_DIR)
    match = profiles_mod.select_profile(profile_list, prelim_full_text, len(probes), dominant_layout)

    sub_doc_by_page = assign_sub_documents(page_order, prelim_text_by_page, match.profile)
    clause_sub_document = match.profile.get("expected_invariants", {}).get("clause_sequence_scope")

    nodes, page_raw_text, tree_quality_flags = build_tree(
        pages_blocks, layout_type_by_page, page_order, sub_doc_by_page, clause_sub_document
    )

    for page, tblocks in table_blocks_by_page.items():
        flat = "\n".join(" | ".join(cell or "" for cell in row) for t in tblocks for row in t.rows)
        page_raw_text[page] = (page_raw_text.get(page, "") + "\n" + flat).strip()

    label_index = entities_mod.build_label_index(nodes)
    entities_mod.resolve_refs_out(nodes, label_index)
    entities_mod.tag_modality(nodes)

    full_text = "\n\n".join(page_raw_text.get(p, "") for p in page_order)
    document_status = guess_document_status(full_text)

    doc_entities = entities_mod.extract_document_entities(nodes, full_text)
    core = core_fields.resolve_core(full_text, document_status)

    for n in nodes:
        n.sub_document = sub_doc_by_page.get(n.pages[0]) if n.pages else None

    tables = build_table_entries(table_blocks_by_page, label_index)

    quality = run_validation(
        str(pdf_path), nodes, page_raw_text, layout_type_by_page, core, match.profile, tree_quality_flags,
    )
    quality = _neutralize_oracle_check(quality)

    with fitz.open(str(pdf_path)) as doc:
        pdf_metadata = {k: str(v) for k, v in (doc.metadata or {}).items() if v}

    all_confidences = [s["mean_confidence"] for s in ocr_stats.values() if s["words_kept"]]
    mean_conf = round(sum(all_confidences) / len(all_confidences), 2) if all_confidences else 0.0

    pages_out = []
    for p in probes:
        raw_text = page_raw_text.get(p.page, "")
        stats = ocr_stats[p.page]
        pages_out.append(
            {
                "page": p.page,
                "page_label": extract_page_label(raw_text) if raw_text else None,
                "sub_document": sub_doc_by_page.get(p.page),
                "width": p.width,
                "height": p.height,
                "rotation": p.rotation,
                "extraction_method": "ocr",
                "route_reason": (
                    f"tesseract lang={lang} psm={psm}+{secondary_psm} dpi={dpi} "
                    f"mean_conf={stats['mean_confidence']} words={stats['words_kept']} "
                    f"pass2_recovered={stats['words_recovered_pass2']} "
                    f"pass2_corrected={stats['words_corrected_pass2']} "
                    f"deskew={stats['deskew_deg']}deg"
                ),
                "layout_type": layout_type_by_page[p.page],
                "has_ruling_lines": p.ruling_line_count > 0,
                "raw_text": raw_text,
                "char_count": len(raw_text),
            }
        )

    structure_out = [asdict(n) for n in nodes]

    try:
        tesseract_version = str(pytesseract.get_tesseract_version())
    except Exception:  # pragma: no cover - version probing is best-effort
        tesseract_version = "unknown"

    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file": pdf_path.name,
            "sha256": sha256_of(pdf_path),
            "page_count": len(probes),
            "pdf_metadata": pdf_metadata,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "1.0.0-ocr",
            "parsers": {
                "primary": f"tesseract {tesseract_version} (pytesseract)",
                "renderer": "PyMuPDF",
                "oracle": "not applicable (OCR source)",
            },
            "ocr": {
                "dpi": dpi,
                "lang": lang,
                "psm": psm,
                "secondary_psm": secondary_psm,
                "words_recovered_pass2": sum(s["words_recovered_pass2"] for s in ocr_stats.values()),
                "words_corrected_pass2": sum(s["words_corrected_pass2"] for s in ocr_stats.values()),
                "min_word_confidence": min_conf,
                "deskew_enabled": deskew,
                "mean_confidence": mean_conf,
                "low_confidence_pages": sorted(
                    p for p, s in ocr_stats.items() if s["words_kept"] and s["mean_confidence"] < 70.0
                ),
                "pages_with_no_text": sorted(p for p, s in ocr_stats.items() if not s["words_kept"]),
                "per_page": {str(p): s for p, s in sorted(ocr_stats.items())},
            },
        },
        "profile": {
            "profile_id": match.profile["profile_id"],
            "match_score": round(match.score, 3),
            "description": match.profile.get("description", ""),
        },
        "core": core,
        "pages": pages_out,
        "structure": structure_out,
        "tables": tables,
        "entities": doc_entities,
        "quality": quality,
    }
    return document


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a scanned/image-only Indonesian contract PDF into raw_extraction.json via OCR"
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF")
    parser.add_argument("--out", type=Path, default=Path("output_ocr"), help="Output directory (default: output_ocr/)")
    parser.add_argument("--profile-dir", type=Path, default=None, help="Override the profile directory")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Render DPI (default: {DEFAULT_DPI})")
    parser.add_argument("--lang", default=DEFAULT_LANG, help=f"Tesseract language(s) (default: {DEFAULT_LANG})")
    parser.add_argument("--psm", type=int, default=DEFAULT_PSM, help=f"Primary Tesseract page segmentation mode (default: {DEFAULT_PSM})")
    parser.add_argument("--secondary-psm", type=int, default=DEFAULT_SECONDARY_PSM, help=f"Recovery-pass PSM for tokens the primary pass drops (default: {DEFAULT_SECONDARY_PSM})")
    parser.add_argument("--single-pass", action="store_true", help="Disable the recovery pass (faster, drops gutter labels)")
    parser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF, help=f"Drop words below this confidence (default: {DEFAULT_MIN_CONF})")
    parser.add_argument("--no-deskew", action="store_true", help="Skip deskew correction")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Write preprocessed page images here for inspection")
    parser.add_argument("--tesseract-cmd", type=Path, default=None, help="Path to tesseract.exe if it is not on PATH")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"error: {args.pdf_path} does not exist", file=sys.stderr)
        return 1

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = str(args.tesseract_cmd)

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        print(
            f"error: tesseract binary not usable ({exc}).\n"
            "Install Tesseract and the 'ind' language data, then either add it to PATH "
            "or pass --tesseract-cmd \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    document = run_ocr_pipeline(
        args.pdf_path, args.out, args.profile_dir,
        dpi=args.dpi, lang=args.lang, psm=args.psm,
        secondary_psm=None if args.single_pass else args.secondary_psm,
        min_conf=args.min_conf, deskew=not args.no_deskew, debug_dir=args.debug_dir,
    )

    out_path = args.out / "raw_extraction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    q = document["quality"]
    ocr_info = document["source"]["ocr"]
    core_status = document["core"]["_status"]
    print(f"profile: {document['profile']['profile_id']} (score={document['profile']['match_score']})")
    print(f"pages: {document['source']['page_count']}  nodes: {len(document['structure'])}  tables: {len(document['tables'])}")
    print(f"ocr: mean_confidence={ocr_info['mean_confidence']}  pass2_recovered={ocr_info['words_recovered_pass2']}  pass2_corrected={ocr_info['words_corrected_pass2']}")
    print(f"     low_conf_pages={ocr_info['low_confidence_pages']}  empty_pages={ocr_info['pages_with_no_text']}")
    print(f"core fields populated: {core_status['fields_populated']}/6  overall_confidence={core_status['overall_confidence']}")
    print(f"validation: {q['pipeline_status']}  hard_fails={q['hard_fail_count']}  warns={q['warn_count']}")
    for c in q["checks"]:
        if c["status"] in ("fail", "warn"):
            print(f"  [{c['status'].upper()}] {c['check']}: {c['detail']}")
    print(f"wrote {out_path}")
    return 0 if q["pipeline_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
