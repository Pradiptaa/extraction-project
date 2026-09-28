"""Stage 4 — Layout Segmentation."""
from __future__ import annotations

from dataclasses import dataclass

from .probe import PageProbe

BIN_COUNT = 50
DOMINANT_MODE_MIN_SHARE = 0.40
LEFT_MODE_MAX_START_FRAC = 0.25
MIN_COLUMN_GAP_FRACTION = 0.15
RULING_LINE_GRID_THRESHOLD = 20
BLANK_CHAR_THRESHOLD = 50


GAP_COVERAGE_TOLERANCE = 0.01 
MIN_GAP_WIDTH_FRACTION = 0.02 
MIN_PARALLEL_SIDE_SHARE = 0.30 
MIN_GAP_OVER_WORD_SPACING = 3.0
COVERAGE_BIN_COUNT = 80


@dataclass
class LayoutInfo:
    page: int
    layout_type: str 
    column_boundary_frac: float | None 
    detector: str
    right_column_start_frac: float | None = None
    column_role: str | None = None


def find_column_corridor(probe: PageProbe) -> tuple[float, float, float] | None:
    words = probe.words
    if not words or not probe.width:
        return None
    coverage = [0] * COVERAGE_BIN_COUNT
    for word in words:
        first = max(0, min(COVERAGE_BIN_COUNT - 1, int(word["x0"] / probe.width * COVERAGE_BIN_COUNT)))
        last = max(0, min(COVERAGE_BIN_COUNT - 1, int(word["x1"] / probe.width * COVERAGE_BIN_COUNT)))
        for i in range(first, last + 1):
            coverage[i] += 1

    spacings: list[float] = []
    for line in _line_groups(words):
        ordered = sorted(line, key=lambda w: w["x0"])
        spacings += [b["x0"] - a["x1"] for a, b in zip(ordered, ordered[1:]) if b["x0"] > a["x1"]]
    word_spacing = median_line_height(spacings, default=2.0) if spacings else 2.0

    threshold = max(1, int(GAP_COVERAGE_TOLERANCE * len(words)))
    min_bins = max(1, int(MIN_GAP_WIDTH_FRACTION * COVERAGE_BIN_COUNT))
    best: tuple[float, float, float] | None = None
    start = None
    for index in range(COVERAGE_BIN_COUNT + 1):
        empty = index < COVERAGE_BIN_COUNT and coverage[index] <= threshold
        if empty and start is None:
            start = index
        elif not empty and start is not None:
            run_start, run_end = start, index
            start = None
            if run_start == 0 or run_end == COVERAGE_BIN_COUNT or (run_end - run_start) < min_bins:
                continue
            gap_width = (run_end - run_start) / COVERAGE_BIN_COUNT * probe.width
            if gap_width < MIN_GAP_OVER_WORD_SPACING * word_spacing:
                continue
            centre = ((run_start + run_end) / 2.0) / COVERAGE_BIN_COUNT
            split_x = centre * probe.width
            left = sum(1 for w in words if w["x1"] <= split_x)
            right = sum(1 for w in words if w["x0"] >= split_x)
            left_share, right_share = left / len(words), right / len(words)
            if min(left_share, right_share) >= MIN_PARALLEL_SIDE_SHARE:
                if best is None or min(left_share, right_share) > min(best[1], best[2]):
                    best = (centre, left_share, right_share)
    return best


REFERENCE_LINE_HEIGHT = 12.0
LINE_GROUP_TOLERANCE_RATIO = 0.25 


def median_line_height(heights: list[float], default: float = REFERENCE_LINE_HEIGHT) -> float:
    usable = sorted(h for h in heights if h > 0)
    return usable[len(usable) // 2] if usable else default


def median_word_height(words: list[dict], default: float = REFERENCE_LINE_HEIGHT) -> float:
    return median_line_height([w["bottom"] - w["top"] for w in words], default)


def _line_groups(words: list[dict], y_tolerance: float | None = None) -> list[list[dict]]:
    if not words:
        return []
    if y_tolerance is None:
        y_tolerance = LINE_GROUP_TOLERANCE_RATIO * median_word_height(words)
    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = [[ordered[0]]]
    for w in ordered[1:]:
        if abs(w["top"] - lines[-1][-1]["top"]) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def classify_layout(probe: PageProbe, detect_parallel_columns: bool = False) -> LayoutInfo:
    if probe.char_count < BLANK_CHAR_THRESHOLD:
        return LayoutInfo(probe.page, "blank", None, "char_count_below_threshold")

    if probe.ruling_line_count >= RULING_LINE_GRID_THRESHOLD:
        return LayoutInfo(probe.page, "ruled_table", None, "ruling_line_count_grid")

    lines = _line_groups(probe.words)
    if not lines:
        return LayoutInfo(probe.page, "blank", None, "no_lines_found")

    if detect_parallel_columns:
        corridor = find_column_corridor(probe)
        if corridor is not None:
            centre, left_share, right_share = corridor
            return LayoutInfo(
                probe.page, "two_column", centre,
                f"parallel_columns(left={left_share:.2f},right={right_share:.2f})",
                right_column_start_frac=centre,
                column_role="parallel",
            )

    x0_fracs = [min(min(w["x0"] for w in line) / probe.width, 1.0) for line in lines]
    bins = [0] * BIN_COUNT
    for f in x0_fracs:
        idx = min(BIN_COUNT - 1, int(f * BIN_COUNT))
        bins[idx] += 1

    total_lines = len(x0_fracs)
    non_empty_bins = [i for i, c in enumerate(bins) if c > 0]
    if not non_empty_bins:
        return LayoutInfo(probe.page, "single_column", None, "single_diffuse_mode")

    modes: list[tuple[float, float, int]] = [] 
    i = 0
    while i < len(non_empty_bins):
        j = i
        while j + 1 < len(non_empty_bins) and non_empty_bins[j + 1] == non_empty_bins[j] + 1:
            j += 1
        count = sum(bins[non_empty_bins[k]] for k in range(i, j + 1))
        start_frac = non_empty_bins[i] / BIN_COUNT
        end_frac = (non_empty_bins[j] + 1) / BIN_COUNT
        modes.append((start_frac, end_frac, count))
        i = j + 1

    if len(modes) == 1:
        return LayoutInfo(probe.page, "single_column", None, "single_dominant_mode")

    dominant = max(modes, key=lambda m: m[2])
    dominant_share = dominant[2] / total_lines

    if dominant_share < DOMINANT_MODE_MIN_SHARE:
        return LayoutInfo(probe.page, "form", None, "multiple_comparable_modes")

    left_candidates = [m for m in modes if m[1] <= dominant[0] and m[0] < LEFT_MODE_MAX_START_FRAC]
    left_candidates.sort(key=lambda m: -m[1])
    for cand in left_candidates:
        gap = dominant[0] - cand[1]
        if gap >= MIN_COLUMN_GAP_FRACTION:
            boundary = (cand[1] + dominant[0]) / 2.0
            right_side_fracs = [f for f in x0_fracs if f > cand[1]]
            right_column_start = min(right_side_fracs) if right_side_fracs else dominant[0]
            return LayoutInfo(
                probe.page, "two_column", boundary, "dominant_mode_plus_separated_left_mode",
                right_column_start_frac=right_column_start,
                column_role="gutter_label",
            )

    return LayoutInfo(probe.page, "single_column", None, "single_dominant_mode_no_left_column")
