"""`pipeline.layout` column roles (fix_plan Phase 5).

A two-column page is one of two different things, and they need opposite reading
orders: a *gutter-label* page carries numbers and short titles beside a body,
and is read row by row; a *parallel* page carries two body columns, and each one
must be read whole. Reading a parallel page row-major splices the right column
into the middle of the left one's sentences.
"""
from __future__ import annotations

import unittest

from pipeline.blocks import extract_text_blocks
from pipeline.layout import classify_layout, find_column_corridor
from pipeline.probe import PageProbe

PAGE_W, PAGE_H = 595.0, 842.0


def word(text: str, x0: float, top: float, width: float = 40.0) -> dict:
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "bottom": top + 12.0,
            "size": 10.0, "fontname": "Helvetica"}


def probe(words: list[dict]) -> PageProbe:
    return PageProbe(page=1, width=PAGE_W, height=PAGE_H, rotation=0,
                     char_count=sum(len(w["text"]) for w in words), word_count=len(words),
                     image_count=0, image_coverage=0.0, fonts=["Helvetica"],
                     ruling_line_count=0, words=words)


def _run_of_words(x_start: float, x_end: float, top: float, seed: int) -> list[dict]:
    """Words filling a span, with ragged boundaries like real prose."""
    out, x = [], x_start
    while x < x_end - 20.0:
        width = 24.0 + ((seed * 7 + int(x)) % 5) * 8.0
        width = min(width, x_end - x)
        out.append(word("kata", x, top, width=width))
        x += width + 4.0
    return out


def parallel_page() -> PageProbe:
    """Two body columns of equal weight, 72..280 and 320..530."""
    words = []
    for row in range(12):
        top = 100.0 + row * 16.0
        words += _run_of_words(72.0, 280.0, top, row)
        words += _run_of_words(320.0, 530.0, top, row + 3)
    return probe(words)


def gutter_label_page() -> PageProbe:
    """A narrow label column (a few short numbers) beside a full body."""
    words = []
    for row in range(12):
        top = 100.0 + row * 16.0
        if row % 4 == 0:
            words.append(word(f"{row}.", 72.0, top, width=18.0))
        words += _run_of_words(200.0, 530.0, top, row)
    return probe(words)


class ColumnRoleTests(unittest.TestCase):
    def test_parallel_columns_are_detected(self) -> None:
        info = classify_layout(parallel_page(), detect_parallel_columns=True)
        self.assertEqual((info.layout_type, info.column_role), ("two_column", "parallel"))

    def test_a_label_gutter_is_not_parallel(self) -> None:
        """Its left side holds a few short numbers, not a share of the words."""
        self.assertIsNone(find_column_corridor(gutter_label_page()))
        info = classify_layout(gutter_label_page(), detect_parallel_columns=True)
        self.assertNotEqual(info.column_role, "parallel")

    def test_detection_is_off_unless_asked(self) -> None:
        self.assertIsNone(classify_layout(parallel_page()).column_role)

    def test_a_single_column_page_has_no_corridor(self) -> None:
        words = [w for row in range(12) for w in _run_of_words(72.0, 530.0, 100.0 + row * 16.0, row)]
        self.assertIsNone(find_column_corridor(probe(words)))

    def test_margins_are_not_corridors(self) -> None:
        """Whitespace at the edges is a margin; a corridor has text both sides."""
        words = [w for row in range(10) for w in _run_of_words(250.0, 380.0, 100.0 + row * 16.0, row)]
        self.assertIsNone(find_column_corridor(probe(words)))


class ParallelReadingOrderTests(unittest.TestCase):
    def test_each_column_is_read_whole(self) -> None:
        page = parallel_page()
        layout = classify_layout(page, detect_parallel_columns=True)
        blocks = extract_text_blocks(page, layout)
        columns = [b.column_index for b in blocks]
        # Every left-column block precedes every right-column one.
        self.assertEqual(columns, sorted(columns))
        self.assertEqual(set(columns), {0, 1})
        left = [b for b in blocks if b.column_index == 0]
        self.assertEqual([b.top for b in left], sorted(b.top for b in left))

    def test_gutter_label_page_keeps_row_major_order(self) -> None:
        page = gutter_label_page()
        layout = classify_layout(page, detect_parallel_columns=True)
        blocks = extract_text_blocks(page, layout)
        if layout.layout_type == "two_column":
            # A label is followed by the body beside it, not by the next label.
            self.assertNotEqual([b.column_index for b in blocks], sorted(b.column_index for b in blocks))


if __name__ == "__main__":
    unittest.main()
