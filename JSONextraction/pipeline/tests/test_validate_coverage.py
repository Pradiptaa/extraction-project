from __future__ import annotations

import unittest
from unittest import mock

from pipeline import validate
from pipeline.probe import PageProbe
from pipeline.router import RouteDecision


def _probe(page: int, nonspace: int) -> PageProbe:
    return PageProbe(page=page, width=595.0, height=842.0, rotation=0, char_count=nonspace, word_count=0,
                     image_count=0, image_coverage=0.0, fonts=["F"], ruling_line_count=0,
                     nonspace_char_count=nonspace)


class RawCharCoverageTest(unittest.TestCase):
    def test_full_coverage_passes(self):
        check = validate.check_raw_char_coverage([_probe(1, 40)], {1: "a" * 40})
        self.assertEqual(check["status"], "pass")

    def test_table_separators_and_whitespace_are_not_counted(self):
        check = validate.check_raw_char_coverage([_probe(1, 40)], {1: " | ".join(["a" * 10] * 4) + "\n  "})
        self.assertEqual(check["status"], "pass")

    def test_lost_text_fails_and_partial_loss_warns(self):
        self.assertEqual(validate.check_raw_char_coverage([_probe(1, 100)], {1: "a" * 40})["status"], "fail")
        self.assertEqual(validate.check_raw_char_coverage([_probe(1, 100)], {1: "a" * 70})["status"], "warn")

    def test_page_number_only_pages_are_ignored(self):
        check = validate.check_raw_char_coverage([_probe(1, 40), _probe(2, 4)], {1: "a" * 40})
        self.assertEqual(check["status"], "pass")

    def test_without_probes_or_native_text_it_skips(self):
        self.assertEqual(validate.check_raw_char_coverage(None, {})["status"], "skip")
        self.assertEqual(validate.check_raw_char_coverage([_probe(1, 0)], {})["status"], "skip")


class RouteCoverageTest(unittest.TestCase):
    def test_statuses(self):
        native = RouteDecision(page=1, method="native", reason="")
        review = RouteDecision(page=2, method="hybrid_review", reason="")
        ocr = RouteDecision(page=3, method="ocr_needed", reason="")
        self.assertEqual(validate.check_route_coverage([native])["status"], "pass")
        self.assertEqual(validate.check_route_coverage([native, review])["status"], "warn")
        self.assertEqual(validate.check_route_coverage([native, review, ocr])["status"], "fail")
        self.assertEqual(validate.check_route_coverage(None)["status"], "skip")

    def test_hybrid_review_fails_only_on_image_covered_pages(self):
        decisions = [RouteDecision(page=1, method="hybrid_review", reason=""),
                     RouteDecision(page=2, method="hybrid_review", reason="")]
        sparse, imaged = _probe(1, 4), _probe(2, 30)
        imaged.image_coverage = 0.95
        self.assertEqual(validate.check_route_coverage(decisions[:1], [sparse])["status"], "warn")
        check = validate.check_route_coverage(decisions, [sparse, imaged])
        self.assertEqual(check["status"], "fail")
        self.assertIn("unhandled_pages=['p2']", check["detail"])

    def test_enforcement_switch_sets_severity(self):
        decision = [RouteDecision(page=1, method="ocr_needed", reason="")]
        self.assertEqual(validate.check_route_coverage(decision)["severity"], "hard_fail")
        with mock.patch.object(validate, "ENFORCE_SOURCE_COVERAGE", False):
            self.assertEqual(validate.check_route_coverage(decision)["severity"], "warn")


class OracleTokenRecallTest(unittest.TestCase):
    def test_cell_order_does_not_matter(self):
        self.assertEqual(validate._token_recall("b | a\nc", "a b c"), 1.0)

    def test_missing_tokens_lower_recall(self):
        self.assertAlmostEqual(validate._token_recall("a b", "a b c d"), 0.5)


class EmptyOutputTest(unittest.TestCase):
    def test_zero_nodes_on_non_blank_document_fails(self):
        self.assertEqual(validate.check_nonempty_tree([], {1: "single_column"})["status"], "fail")
        self.assertEqual(validate.check_nonempty_tree([], {1: "blank"})["status"], "pass")

    def test_no_countable_pages(self):
        self.assertEqual(validate.check_char_conservation([], {1: ""}, {1: "blank"})["status"], "fail")
        self.assertEqual(validate.check_char_conservation([], {1: "a | b"}, {1: "ruled_table"})["status"], "warn")

    def test_summary_counts_only_hard_fail_failures(self):
        checks = [validate._check("x", "fail", "", "warn"), validate._check("y", "warn", "", "hard_fail")]
        self.assertEqual(validate.summarize_quality(checks, [])["pipeline_status"], "passed")
        checks.append(validate._check("z", "fail", "", "hard_fail"))
        summary = validate.summarize_quality(checks, [])
        self.assertEqual((summary["pipeline_status"], summary["hard_fail_count"], summary["warn_count"]), ("failed", 1, 1))


class OcrOracleNeutralizationTest(unittest.TestCase):
    def test_neutralized_oracle_failure_does_not_fail_ocr_runs(self):
        from pipeline.ocr_main import _neutralize_oracle_check
        checks = [validate._check("dual_parser_oracle", "fail", "", "hard_fail"),
                  validate._check("raw_char_coverage", "skip", "", "info")]
        quality = _neutralize_oracle_check(validate.summarize_quality(checks, []))
        self.assertEqual(quality["pipeline_status"], "passed")
        self.assertEqual(quality["checks"][0]["status"], "skip")


if __name__ == "__main__":
    unittest.main()
