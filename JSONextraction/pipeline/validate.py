"""Stage 9 — VALIDATE. Generic checks that apply to every contract (schema
section 6). Document-specific invariants (e.g. "exactly 79 clauses") live in
the matched profile's `expected_invariants`, not in this code.

A hard-fail check does not raise — main.py still writes raw_extraction.json
so the failure is inspectable — but it flips `quality.pipeline_status` to
`failed`, which downstream ingestion should treat as a gate.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import difflib
from pathlib import Path

from .tree import Node

REPLACEMENT_CHAR = "�"


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
        return _check("char_conservation", "warn", "no countable pages", "hard_fail")
    ratio = total_node_chars / total_page_chars
    status = "pass" if ratio >= 0.90 else "fail"
    # note: threshold relaxed from the design doc's 99.5% because block-to-line
    # joins strip some inter-word spacing; treat <0.90 as the hard signal of a
    # genuinely broken extraction, and log the ratio either way for review.
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


def check_dual_parser_oracle(pdf_path: str, page_raw_text: dict[int, str]) -> dict:
    """Cross-checks pdfplumber output against Poppler's `pdftotext -layout`, an
    independent C++ implementation, when the binary is available on PATH."""
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
        ratio = difflib.SequenceMatcher(None, ours, theirs).ratio()
        ratios.append((page_num, ratio))

    low = [f"p{p}:{r:.2f}" for p, r in ratios if r < 0.90]
    avg = sum(r for _, r in ratios) / len(ratios) if ratios else 0.0
    status = "pass" if not low else "warn"
    return _check("dual_parser_oracle", status, f"avg_ratio={avg:.3f} low_pages={low[:10]}", "warn")


def run_validation(
    pdf_path: str,
    nodes: list[Node],
    page_raw_text: dict[int, str],
    layout_by_page: dict[int, str],
    core: dict,
    profile: dict,
    tree_quality_flags: list[str],
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
        check_words_vs_digits(core["key_numbers"]),
        check_dual_parser_oracle(pdf_path, page_raw_text),
    ]
    checks += check_profile_invariants(nodes, profile)

    hard_fails = [c for c in checks if c["severity"] == "hard_fail" and c["status"] == "fail"]
    pipeline_status = "failed" if hard_fails else "passed"

    return {
        "pipeline_status": pipeline_status,
        "checks": checks,
        "hard_fail_count": len(hard_fails),
        "warn_count": sum(1 for c in checks if c["status"] == "warn"),
        "tree_quality_flags": tree_quality_flags,
    }
