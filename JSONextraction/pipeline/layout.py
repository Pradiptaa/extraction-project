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


@dataclass
class LayoutInfo:
    page: int
    layout_type: str          # single_column | two_column | ruled_table | form | mixed | blank
    column_boundary_frac: float | None   # fraction of page width, only for two_column
    detector: str
    right_column_start_frac: float | None = None


def _line_groups(words: list[dict], y_tolerance: float = 3.0) -> list[list[dict]]:
    """Cluster words into lines by `top` proximity."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = [[ordered[0]]]
    for w in ordered[1:]:
        if abs(w["top"] - lines[-1][-1]["top"]) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def classify_layout(probe: PageProbe) -> LayoutInfo:
    if probe.char_count < BLANK_CHAR_THRESHOLD:
        return LayoutInfo(probe.page, "blank", None, "char_count_below_threshold")

    if probe.ruling_line_count >= RULING_LINE_GRID_THRESHOLD:
        return LayoutInfo(probe.page, "ruled_table", None, "ruling_line_count_grid")

    lines = _line_groups(probe.words)
    if not lines:
        return LayoutInfo(probe.page, "blank", None, "no_lines_found")

    # Leftmost x0 per line, not line[0]: a left-column heading can sit a fraction
    # of a point lower than the right-column text beside it.
    x0_fracs = [min(min(w["x0"] for w in line) / probe.width, 1.0) for line in lines]
    bins = [0] * BIN_COUNT
    for f in x0_fracs:
        idx = min(BIN_COUNT - 1, int(f * BIN_COUNT))
        bins[idx] += 1

    total_lines = len(x0_fracs)
    non_empty_bins = [i for i, c in enumerate(bins) if c > 0]
    if not non_empty_bins:
        return LayoutInfo(probe.page, "single_column", None, "single_diffuse_mode")

    modes: list[tuple[float, float, int]] = []  # (start_frac, end_frac, count)
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
            # Hanging indents make the bin-run start unreliable; use the leftmost
            # line actually past the gap.
            right_side_fracs = [f for f in x0_fracs if f > cand[1]]
            right_column_start = min(right_side_fracs) if right_side_fracs else dominant[0]
            return LayoutInfo(
                probe.page, "two_column", boundary, "dominant_mode_plus_separated_left_mode",
                right_column_start_frac=right_column_start,
            )

    return LayoutInfo(probe.page, "single_column", None, "single_dominant_mode_no_left_column")
