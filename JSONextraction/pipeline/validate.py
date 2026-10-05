"""Stage 9 — Validate. """
from __future__ import annotations

import re
import shutil
import subprocess
import difflib
from collections import Counter
from pathlib import Path

from .tree import Node

REPLACEMENT_CHAR = "�"
ENFORCE_SOURCE_COVERAGE = True
RAW_COVERAGE_FAIL_BELOW = 0.5
RAW_COVERAGE_WARN_BELOW = 0.8
RAW_COVERAGE_MIN_CHARS = 20
ORACLE_FAIL_BELOW = 0.5
ORACLE_WARN_BELOW = 0.90
UNREADABLE_IMAGE_COVERAGE = 0.80
_UNCOUNTED_RE = re.compile(r"[\s|]")


def _coverage_severity() -> str:
    return "hard_fail" if ENFORCE_SOURCE_COVERAGE else "warn"


def _check(name: str, status: str, detail: str, severity: str) -> dict:
    return {"check": name, "status": status, "severity": severity, "detail": detail}


def check_core_presence(core: dict) -> dict:
    required = {"document_type", "contract_name", "contract_number", "parties", "key_dates", "key_numbers", "_status"}
    missing = required - set(core.keys())
    if missing:
        return _check("core_presence", "fail", f"missing keys: {sorted(missing)}", "hard_fail")
    return _check("core_presence", "pass", "all 6 core fields + _status present", "hard_fail")


def check_char_conservation(nodes: list[Node], page_raw_text: dict[int, str], layout_by_page: dict[int, str]) -> dict:
    countable_pages = {p for p, lt in layout_by_page.items() if lt not in ("ruled_table", "blank")}
    total_page_chars = sum(len(page_raw_text.get(p, "")) for p in countable_pages)
    total_node_chars = sum(
        len(n.text_raw or "") for n in nodes if n.pages and any(p in countable_pages for p in n.pages)
    )
    if total_page_chars == 0:
        if any(text.strip() for text in page_raw_text.values()):
            return _check("char_conservation", "warn", "no countable pages, text only in tables", "hard_fail")
        return _check("char_conservation", "fail", "no countable pages and no extracted text", "hard_fail")
    ratio = total_node_chars / total_page_chars
    status = "pass" if ratio >= 0.90 else "fail"
    return _check("char_conservation", status, f"ratio={ratio:.4f}", "hard_fail")


def check_no_duplicate_spans(nodes: list[Node]) -> dict:
    seen = set()
    dupes = 0
    for n in nodes:
        if not n.char_span:
            continue
        key = (n.char_span.get("page"), n.char_span.get("start"), n.char_span.get("end"))
        if key in seen:
            dupes += 1
        seen.add(key)
    status = "pass" if dupes == 0 else "fail"
    return _check("no_duplicate_spans", status, f"duplicate_spans={dupes}", "hard_fail")


def check_tree_integrity(nodes: list[Node]) -> dict:
    by_id = {n.node_id: n for n in nodes}
    orphans = [n.node_id for n in nodes if n.parent_id and n.parent_id not in by_id]
    depth_violations = [
        n.node_id for n in nodes if n.parent_id and n.parent_id in by_id and by_id[n.parent_id].depth >= n.depth
    ]
    if orphans or depth_violations:
        return _check(
            "tree_integrity", "fail",
            f"orphans={orphans[:5]} depth_violations={depth_violations[:5]}", "hard_fail",
        )
    return _check("tree_integrity", "pass", f"{len(nodes)} nodes, no orphans or depth violations", "hard_fail")


def check_sibling_sequence(tree_quality_flags: list[str]) -> dict:
    breaks = [f for f in tree_quality_flags if f.startswith("sequence_break")]
    status = "pass" if not breaks else "warn"
    return _check("sibling_sequence", status, f"{len(breaks)} sequence breaks", "warn")


def check_identifier_survival(contract_number_core: dict, page_raw_text: dict[int, str]) -> dict:
    value = contract_number_core.get("value")
    if not value:
        return _check("identifier_survival", "warn", "contract_number not resolved, nothing to verify", "hard_fail")
    full_text = "\n".join(page_raw_text.values())
    found = value in full_text
    status = "pass" if found else "fail"
    return _check("identifier_survival", status, f"contract_number verbatim survival: {found}", "hard_fail")


def check_encoding_sanity(page_raw_text: dict[int, str]) -> dict:
    count = sum(text.count(REPLACEMENT_CHAR) for text in page_raw_text.values())
    status = "pass" if count == 0 else "fail"
    return _check("encoding_sanity", status, f"replacement_char_count={count}", "hard_fail")


def check_placeholder_tagging(nodes: list[Node]) -> dict:
    placeholder_pattern = re.compile(r"…{1,}|\.{4,}|\[[^\]]{1,80}\]")
    untagged = 0
    for n in nodes:
        if placeholder_pattern.search(n.text_raw or ""):
            if "template_placeholder" not in n.extraction.get("flags", []):
                untagged += 1
    status = "pass" if untagged == 0 else "warn"
    return _check("placeholder_tagging", status, f"untagged_placeholder_nodes={untagged}", "warn")


_HEADING_BEARING_TYPES = ("clause", "article", "section", "part")
_ALLCAPS_RUN_RE = re.compile(r"^[A-Z][A-Z0-9 ,./()\-]{7,}")


def check_title_bleed(nodes: list[Node]) -> dict:
    bled = [
        n for n in nodes
        if n.node_type in _HEADING_BEARING_TYPES
        and not (n.title or "").strip()
        and _ALLCAPS_RUN_RE.match((n.text_raw or "").strip())
    ]
    detail = f"nodes_with_untitled_allcaps_body={len(bled)}"
    if bled:
        detail += " e.g. " + ", ".join(f"{n.node_id}:{(n.text_raw or '')[:40]!r}" for n in bled[:3])
    return _check("title_bleed", "pass" if not bled else "warn", detail, "warn")


def check_words_vs_digits(key_numbers_core: dict) -> dict:
    mismatches = [n for n in key_numbers_core.get("value", []) if n.get("words_check") == "mismatch"]
    status = "pass" if not mismatches else "warn"
    return _check("words_vs_digits", status, f"mismatches={len(mismatches)}", "warn")


def check_profile_invariants(nodes: list[Node], profile: dict) -> list[dict]:
    invariants = profile.get("expected_invariants", {})
    results = []
    scope = invariants.get("clause_sequence_scope")
    clause_labels = [
        n.label_normalized
        for n in nodes
        if n.node_type == "clause" and n.label_normalized and (scope is None or n.sub_document == scope)
    ]

    if invariants.get("clause_sequence_gapless"):
        nums = sorted(int(x) for x in clause_labels if x.isdigit())
        gapless = nums == list(range(1, len(nums) + 1)) if nums else False
        results.append(
            _check(
                "profile:clause_sequence_gapless",
                "pass" if gapless else "warn",
                f"clause_count={len(nums)} sequence={'gapless' if gapless else 'has gaps or duplicates'}",
                "warn",
            )
        )

    if "min_clauses" in invariants:
        ok = len(clause_labels) >= invariants["min_clauses"]
        results.append(
            _check(
                "profile:min_clauses",
                "pass" if ok else "warn",
                f"clause_count={len(clause_labels)} min_required={invariants['min_clauses']}",
                "warn",
            )
        )

    if "exact_clause_count" in invariants:
        ok = len(clause_labels) == invariants["exact_clause_count"]
        results.append(
            _check(
                "profile:exact_clause_count",
                "pass" if ok else "warn",
                f"clause_count={len(clause_labels)} expected={invariants['exact_clause_count']}",
                "warn",
            )
        )

    return results


def _token_recall(ours: str, theirs: str) -> float:
    ours_bag = Counter(_UNCOUNTED_RE.sub(" ", ours).split())
    theirs_bag = Counter(_UNCOUNTED_RE.sub(" ", theirs).split())
    total = sum(theirs_bag.values())
    return sum((ours_bag & theirs_bag).values()) / total if total else 1.0


def check_dual_parser_oracle(pdf_path: str, page_raw_text: dict[int, str],
                             layout_by_page: dict[int, str] | None = None) -> dict:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return _check("dual_parser_oracle", "skip", "pdftotext (poppler) not found on PATH", "info")

    try:
        proc = subprocess.run(
            [pdftotext, "-layout", str(pdf_path), "-"],
            capture_output=True, timeout=60, check=True,
        )
    except Exception as exc: 
        return _check("dual_parser_oracle", "skip", f"pdftotext invocation failed: {exc}", "info")

    stdout_text = proc.stdout.decode("utf-8", errors="replace")
    poppler_pages = stdout_text.split("\f")

    ratios = []
    for page_num, our_text in page_raw_text.items():
        if page_num - 1 >= len(poppler_pages):
            continue
        theirs = re.sub(r"\s+", " ", poppler_pages[page_num - 1]).strip()
        ours = re.sub(r"\s+", " ", our_text).strip()
        if not theirs and not ours:
            continue
        if len(_UNCOUNTED_RE.sub("", theirs)) < RAW_COVERAGE_MIN_CHARS:
            continue
        if (layout_by_page or {}).get(page_num) == "ruled_table":
            ratio = _token_recall(ours, theirs)
        else:
            ratio = difflib.SequenceMatcher(None, ours, theirs).ratio()
        ratios.append((page_num, ratio))

    failing = [f"p{p}:{r:.2f}" for p, r in ratios if r < ORACLE_FAIL_BELOW]
    low = [f"p{p}:{r:.2f}" for p, r in ratios if ORACLE_FAIL_BELOW <= r < ORACLE_WARN_BELOW]
    avg = sum(r for _, r in ratios) / len(ratios) if ratios else 0.0
    status = "fail" if failing else "warn" if low else "pass"
    detail = f"avg_ratio={avg:.3f} failing_pages={failing[:10]} low_pages={low[:10]}"
    return _check("dual_parser_oracle", status, detail, _coverage_severity())


def check_nonempty_tree(nodes: list[Node], layout_by_page: dict[int, str]) -> dict:
    content_pages = [p for p, lt in layout_by_page.items() if lt != "blank"]
    if nodes or not content_pages:
        return _check("nonempty_tree", "pass", f"{len(nodes)} nodes over {len(content_pages)} non-blank pages", "hard_fail")
    return _check("nonempty_tree", "fail", f"0 nodes on {len(content_pages)} non-blank pages", "hard_fail")


def summarize_quality(checks: list[dict], tree_quality_flags: list[str]) -> dict:
    hard_fails = [c for c in checks if c["severity"] == "hard_fail" and c["status"] == "fail"]
    return {
        "pipeline_status": "failed" if hard_fails else "passed",
        "checks": checks,
        "hard_fail_count": len(hard_fails),
        "warn_count": sum(1 for c in checks if c["status"] == "warn"),
        "tree_quality_flags": tree_quality_flags,
    }


def check_raw_char_coverage(probes: list | None, page_raw_text: dict[int, str]) -> dict:
    if probes is None:
        return _check("raw_char_coverage", "skip", "no native text layer to compare against", "info")
    ratios = []
    for p in probes:
        raw = p.nonspace_char_count
        if not raw or raw < RAW_COVERAGE_MIN_CHARS:
            continue
        extracted = len(_UNCOUNTED_RE.sub("", page_raw_text.get(p.page, "")))
        ratios.append((p.page, extracted / raw))
    if not ratios:
        return _check("raw_char_coverage", "skip", "no page has native characters", "info")
    failing = [f"p{p}:{r:.2f}" for p, r in ratios if r < RAW_COVERAGE_FAIL_BELOW]
    low = [f"p{p}:{r:.2f}" for p, r in ratios if RAW_COVERAGE_FAIL_BELOW <= r < RAW_COVERAGE_WARN_BELOW]
    status = "fail" if failing else "warn" if low else "pass"
    worst = min(ratios, key=lambda pr: pr[1])
    detail = f"min=p{worst[0]}:{worst[1]:.2f} failing_pages={failing[:10]} low_pages={low[:10]}"
    return _check("raw_char_coverage", status, detail, _coverage_severity())


def check_route_coverage(route_decisions: list | None, probes: list | None = None) -> dict:
    if route_decisions is None:
        return _check("route_coverage", "skip", "no route decisions supplied", "info")
    coverage = {p.page: p.image_coverage for p in probes or []}
    unhandled = [
        f"p{r.page}" for r in route_decisions
        if r.method == "ocr_needed"
        or (r.method == "hybrid_review" and coverage.get(r.page, 0.0) >= UNREADABLE_IMAGE_COVERAGE)
    ]
    review = [f"p{r.page}" for r in route_decisions if r.method == "hybrid_review" and f"p{r.page}" not in unhandled]
    status = "fail" if unhandled else "warn" if review else "pass"
    detail = f"unhandled_pages={unhandled[:10]} hybrid_review_pages={review[:10]} of {len(route_decisions)}"
    return _check("route_coverage", status, detail, _coverage_severity())


def run_validation(
    pdf_path: str,
    nodes: list[Node],
    page_raw_text: dict[int, str],
    layout_by_page: dict[int, str],
    core: dict,
    profile: dict,
    tree_quality_flags: list[str],
    probes: list | None = None,
    route_decisions: list | None = None,
) -> dict:
    checks = [
        check_core_presence(core),
        check_char_conservation(nodes, page_raw_text, layout_by_page),
        check_no_duplicate_spans(nodes),
        check_tree_integrity(nodes),
        check_sibling_sequence(tree_quality_flags),
        check_identifier_survival(core["contract_number"], page_raw_text),
        check_encoding_sanity(page_raw_text),
        check_placeholder_tagging(nodes),
        check_title_bleed(nodes),
        check_words_vs_digits(core["key_numbers"]),
        check_dual_parser_oracle(pdf_path, page_raw_text, layout_by_page),
        check_raw_char_coverage(probes, page_raw_text),
        check_route_coverage(route_decisions, probes),
        check_nonempty_tree(nodes, layout_by_page),
    ]
    checks += check_profile_invariants(nodes, profile)
    return summarize_quality(checks, tree_quality_flags)
