"""Scores raw_extraction.json against hand-verified ground truth.

Two independent checks, run separately because they answer different
questions:

  1. Core-field + structural-invariant accuracy, against
     ground_truth/*.ground_truth.json — "did extraction get the six
     guaranteed fields and the document-level facts right?"
  2. Node-level text accuracy, against an annotated review CSV produced by
     sample_review.py and filled in by a human — "is the tree's text
     faithful to the source PDF, sampled across the whole document?"

The internal `quality` block already in raw_extraction.json (character
conservation, tree integrity, ...) checks the extraction is *self-consistent*.
This script checks it is *correct*, which self-consistency cannot prove.

Usage:
    python -m pipeline.evaluate output/raw_extraction.json \
        --ground-truth ground_truth/rancangan_kontrak1.ground_truth.json \
        --review-csv review/sample_for_review.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path
from typing import Any


class Result:
    def __init__(self, name: str, passed: bool, detail: str, status: str | None = None) -> None:
        self.name = name
        self.passed = passed
        self.detail = detail
        # AMBIGUOUS/NOT_FOUND are distinct from FAIL: they mean the check
        # couldn't resolve to a single node to examine at all (a locate that
        # matched 0 or >1 nodes), not that the node's content was wrong.
        # Both still count as failures for the overall pass/fail roll-up.
        self.status = status or ("PASS" if passed else "FAIL")

    def line(self) -> str:
        return f"  [{self.status}] {self.name}: {self.detail}"


def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def check_document_type(core: dict, expected: dict) -> Result:
    actual = core["document_type"]["value"]
    actual_subtype = core["document_type"].get("subtype")
    ok = actual == expected["expected_value"] and (
        expected.get("expected_subtype") is None or actual_subtype == expected["expected_subtype"]
    )
    return Result("core.document_type", ok, f"expected={expected['expected_value']}/{expected.get('expected_subtype')} actual={actual}/{actual_subtype}")


def check_contract_name(core: dict, expected: dict) -> Result:
    actual = core["contract_name"]["value"] or ""
    ratio = fuzzy_ratio(actual, expected["expected_value"])
    ok = ratio >= expected.get("threshold", 0.85)
    return Result("core.contract_name", ok, f"expected={expected['expected_value']!r} actual={actual!r} similarity={ratio:.2f}")


def check_contract_number(core: dict, expected: dict) -> Result:
    actual = core["contract_number"]["value"]
    ok = actual == expected["expected_value"]
    return Result("core.contract_number", ok, f"expected={expected['expected_value']!r} actual={actual!r}")


def check_parties(core: dict, expected: dict) -> list[Result]:
    results = []
    actual_parties = core["parties"]["value"] or []
    results.append(
        Result(
            "core.parties.count",
            len(actual_parties) == expected["expected_count"],
            f"expected={expected['expected_count']} actual={len(actual_parties)}",
        )
    )
    for check in expected["checks"]:
        candidates = [
            p for p in actual_parties
            if any(term in (p.get("role") or "").lower() or term in (p.get("role_label") or "").lower() for term in check["role_contains"])
        ]
        if not candidates:
            results.append(Result(f"core.parties[{check['description']}]", False, "no matching party found by role"))
            continue
        p = candidates[0]
        rep_name = (p.get("representative") or {}).get("name")
        nip = ((p.get("representative") or {}).get("identifier") or {}).get("value")
        org = (p.get("organization") or {}).get("value")

        exp_name = check.get("expected_representative_name")
        exp_nip = check.get("expected_nip")
        exp_org_contains = check.get("expected_organization_contains")

        name_ok = (exp_name is None and rep_name is None) or (exp_name is not None and rep_name == exp_name)
        nip_ok = (exp_nip is None and nip is None) or (exp_nip is not None and nip == exp_nip)
        org_ok = (exp_org_contains is None and True) or (exp_org_contains is not None and exp_org_contains.lower() in (org or "").lower())

        ok = name_ok and nip_ok and org_ok
        results.append(
            Result(
                f"core.parties[{check['description']}]",
                ok,
                f"name: expected={exp_name!r} actual={rep_name!r} | nip: expected={exp_nip!r} actual={nip!r} | org_contains: {exp_org_contains!r} in {org!r}",
            )
        )
    return results


def check_key_dates(core: dict, expected: dict) -> list[Result]:
    actual_dates = {d.get("date") for d in (core["key_dates"]["value"] or [])}
    results = []
    for exp in expected["expected_values"]:
        ok = exp["expected_date"] in actual_dates
        results.append(Result(f"core.key_dates[{exp['description']}]", ok, f"expected {exp['expected_date']!r} in {sorted(d for d in actual_dates if d)}"))
    return results


def check_key_numbers(core: dict, expected: dict) -> list[Result]:
    """Each actual entry can satisfy at most one expected entry: without this,
    two distinct expectations of the same type (e.g. two penalty rates) can
    both show PASS against the SAME single real match — a false positive
    that hides the pipeline only having found one of the two."""
    actual_numbers = core["key_numbers"]["value"] or []
    used: set[int] = set()
    results = []

    def matches(exp: dict, m: dict) -> bool:
        if m.get("type") != exp["type"]:
            return False
        # Only enforce subtype agreement when the actual entry HAS a subtype
        # field — older pipeline output without subtypes should still match.
        if "subtype" in exp and "subtype" in m and m.get("subtype") != exp["subtype"]:
            return False
        if "expected_amount" in exp:
            return m.get("amount") == exp["expected_amount"]
        if "expected_amount_approx" in exp:
            tol = exp.get("tolerance", 0.0001)
            return m.get("amount") is not None and abs(m["amount"] - exp["expected_amount_approx"]) <= tol
        if "expected_value" in exp:
            return m.get("value") == exp["expected_value"]
        return True

    for exp in expected["expected_values"]:
        label = exp["type"] + (f"/{exp['subtype']}" if "subtype" in exp else "")
        if exp["type"] == "contract_value" and exp.get("expected_amount") is None:
            idx = next((i for i, m in enumerate(actual_numbers) if i not in used and m.get("type") == "contract_value" and m.get("amount") is None), None)
            ok = idx is not None
            if ok:
                used.add(idx)
            results.append(Result("core.key_numbers[contract_value=null]", ok, f"matches={[actual_numbers[idx]] if ok else []}"))
            continue

        idx = next((i for i, m in enumerate(actual_numbers) if i not in used and matches(exp, m)), None)
        ok = idx is not None
        if ok:
            used.add(idx)
        same_type = [m for m in actual_numbers if m.get("type") == exp["type"]]
        detail = f"expected={exp} actual_match={actual_numbers[idx]}" if ok else f"expected={exp} candidates_of_this_type={same_type}"
        results.append(Result(f"core.key_numbers[{label}]", ok, detail))
    return results


def check_structural_invariants(nodes: list[dict], pages: list[dict], expected: dict) -> list[Result]:
    results = []
    if "clause_count_general_terms" in expected:
        exp = expected["clause_count_general_terms"]
        count = sum(1 for n in nodes if n["node_type"] == "clause" and n.get("sub_document") == "general_terms")
        tol = exp.get("tolerance", 0)
        ok = abs(count - exp["expected"]) <= tol
        results.append(Result("structural.clause_count_general_terms", ok, f"expected={exp['expected']}±{tol} actual={count}"))
    if "surat_perjanjian_pasal_count" in expected:
        exp = expected["surat_perjanjian_pasal_count"]
        count = sum(1 for n in nodes if n["node_type"] == "article")
        results.append(Result("structural.surat_perjanjian_pasal_count", count == exp["expected"], f"expected={exp['expected']} actual={count}"))
    if "sub_document_count" in expected:
        # Counted from pages[], not structure[] — a sub-document that is
        # entirely tabular (e.g. SSKK, all content inside ruled-table cells)
        # can legitimately produce zero prose nodes, so counting distinct
        # sub_document values on nodes alone would undercount it.
        exp = expected["sub_document_count"]
        count = len({p.get("sub_document") for p in pages if p.get("sub_document")})
        results.append(Result("structural.sub_document_count", count == exp["expected"], f"expected={exp['expected']} actual={count}"))
    return results


def check_identifier_survival(page_texts: str, identifiers: list[str]) -> list[Result]:
    return [Result(f"identifier_survival[{ident}]", ident in page_texts, "verbatim substring found" if ident in page_texts else "NOT FOUND in page text") for ident in identifiers]


def _locate_nodes(nodes: list[dict], locate: dict) -> list[dict]:
    """Every field in `locate` must match for a node to be a candidate.
    Deliberately conjunctive (AND, not OR) and deliberately strict — a
    regression check is only trustworthy if it's known to be examining the
    right node, so under-specifying `locate` should surface as AMBIGUOUS,
    never as a silent match against whichever node happened to come first."""
    matches = []
    for n in nodes:
        if "sub_document" in locate and n.get("sub_document") != locate["sub_document"]:
            continue
        if "node_type" in locate and n.get("node_type") != locate["node_type"]:
            continue
        if "label_normalized" in locate and n.get("label_normalized") != locate["label_normalized"]:
            continue
        if "pages_contains" in locate and locate["pages_contains"] not in (n.get("pages") or []):
            continue
        if "text_raw_contains" in locate and locate["text_raw_contains"] not in (n.get("text_raw") or ""):
            continue
        if "text_raw_equals" in locate and n.get("text_raw") != locate["text_raw_equals"]:
            continue
        if "title_contains" in locate and locate["title_contains"] not in (n.get("title") or ""):
            continue
        matches.append(n)
    return matches


def _check_node_expect(node: dict, expect: dict, by_id: dict[str, dict]) -> list[str]:
    """Returns failure-reason strings; an empty list means the node satisfies
    every assertion in `expect`."""
    failures = []
    if "node_type_equals" in expect and node.get("node_type") != expect["node_type_equals"]:
        failures.append(f"node_type_equals: expected {expect['node_type_equals']!r} actual {node.get('node_type')!r}")
    if expect.get("title_is_none") is True and node.get("title") is not None:
        failures.append(f"title_is_none: expected None actual {node.get('title')!r}")
    if "title_equals" in expect and node.get("title") != expect["title_equals"]:
        failures.append(f"title_equals: expected {expect['title_equals']!r} actual {node.get('title')!r}")
    if "title_contains" in expect and expect["title_contains"] not in (node.get("title") or ""):
        failures.append(f"title_contains: expected substring {expect['title_contains']!r} not in {node.get('title')!r}")
    if "text_raw_equals" in expect and node.get("text_raw") != expect["text_raw_equals"]:
        failures.append(f"text_raw_equals: expected {expect['text_raw_equals']!r} actual {node.get('text_raw')!r}")
    if "text_raw_contains" in expect and expect["text_raw_contains"] not in (node.get("text_raw") or ""):
        failures.append(f"text_raw_contains: expected substring {expect['text_raw_contains']!r} not in {node.get('text_raw')!r}")
    if "has_child" in expect:
        want = expect["has_child"]
        child_ids = node.get("children") or []
        found = any(
            child is not None
            and (want.get("label_normalized") is None or child.get("label_normalized") == want["label_normalized"])
            and (want.get("text_raw_contains") is None or want["text_raw_contains"] in (child.get("text_raw") or ""))
            for child in (by_id.get(cid) for cid in child_ids)
        )
        if not found:
            failures.append(f"has_child: no child among {child_ids} matches {want}")
    return failures


def check_regressions(document: dict, regression_checks: dict) -> list[Result]:
    """Runs the permanent, accumulating checklist of every bug found and
    fixed via the review process — the fixed counterpart to sample_review.py's
    fresh random draw each round. A random sample can prove a NEW bug exists;
    it can't prove an OLD one stayed fixed, because it isn't guaranteed to
    resample the same node twice. These checks are the same every run."""
    nodes = document["structure"]
    by_id = {n["node_id"]: n for n in nodes}
    results = []
    for check in regression_checks.get("checks", []):
        cid = check["id"]
        kind = check.get("kind", "node")
        matches = _locate_nodes(nodes, check["locate"])

        if kind == "count":
            count = len(matches)
            expect = check["expect"]
            ok = True
            if "count_equals" in expect:
                ok = ok and count == expect["count_equals"]
            if "count_lte" in expect:
                ok = ok and count <= expect["count_lte"]
            if "count_gte" in expect:
                ok = ok and count >= expect["count_gte"]
            results.append(Result(f"regression[{cid}]", ok, f"count={count} expect={expect}"))
            continue

        # kind == "node": locate must resolve to exactly one node. 0 or >1
        # matches means the check cannot trust what it would be examining,
        # so it fails loudly as NOT_FOUND/AMBIGUOUS rather than guessing.
        if len(matches) == 0:
            results.append(Result(f"regression[{cid}]", False, f"locate={check['locate']}", status="NOT_FOUND"))
            continue
        if len(matches) > 1:
            ids = [m["node_id"] for m in matches]
            results.append(Result(f"regression[{cid}]", False, f"{len(matches)} nodes matched: {ids}", status="AMBIGUOUS"))
            continue

        failures = _check_node_expect(matches[0], check["expect"], by_id)
        results.append(Result(f"regression[{cid}]", not failures, "OK" if not failures else "; ".join(failures)))
    return results


def evaluate_core(document: dict, gt: dict) -> list[Result]:
    core = document["core"]
    c = gt["core"]
    results = [
        check_document_type(core, c["document_type"]),
        check_contract_name(core, c["contract_name"]),
        check_contract_number(core, c["contract_number"]),
    ]
    results += check_parties(core, c["parties"])
    results += check_key_dates(core, c["key_dates"])
    results += check_key_numbers(core, c["key_numbers"])
    return results


def evaluate_review_csv(csv_path: Path) -> tuple[list[Result], dict[str, Any]]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    judged = [r for r in rows if r.get("correct_yn", "").strip().upper() in ("Y", "N")]
    unjudged = [r for r in rows if r not in judged]
    correct = [r for r in judged if r["correct_yn"].strip().upper() == "Y"]
    incorrect = [r for r in judged if r["correct_yn"].strip().upper() == "N"]

    results = []
    if unjudged:
        results.append(Result("review_csv.completeness", False, f"{len(unjudged)}/{len(rows)} rows not yet judged (correct_yn blank) — fill these in for a real score"))
    if judged:
        accuracy = len(correct) / len(judged)
        results.append(Result("review_csv.node_accuracy", accuracy >= 0.99, f"{len(correct)}/{len(judged)} judged rows correct ({accuracy:.1%}), target >=99%"))
        for r in incorrect[:20]:
            results.append(
                Result(
                    f"review_csv.node[{r.get('node_id')}]",
                    False,
                    f"page={r.get('pages')} label={r.get('label')!r} notes={r.get('notes')!r} corrected_text={r.get('corrected_text')!r}",
                )
            )
    stats = {"total_rows": len(rows), "judged": len(judged), "correct": len(correct), "incorrect": len(incorrect), "unjudged": len(unjudged)}
    return results, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw_extraction.json against hand-verified ground truth")
    parser.add_argument("raw_extraction", type=Path)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, default=None, help="Optional annotated sample from sample_review.py")
    parser.add_argument(
        "--regression-checks",
        type=Path,
        default=Path("ground_truth/regression_checks.json"),
        help="Permanent per-bug checklist (default: ground_truth/regression_checks.json). Pass a nonexistent path or omit the file to skip.",
    )
    args = parser.parse_args()

    with open(args.raw_extraction, encoding="utf-8") as f:
        document = json.load(f)
    with open(args.ground_truth, encoding="utf-8") as f:
        gt = json.load(f)

    all_results: list[Result] = []
    all_results += evaluate_core(document, gt)
    all_results += check_structural_invariants(document["structure"], document["pages"], gt.get("structural_invariants", {}))

    page_texts = "\n".join(p["raw_text"] for p in document["pages"])
    all_results += check_identifier_survival(page_texts, gt.get("identifier_survival_checks", []))

    print("=== Core field & structural evaluation ===")
    for r in all_results:
        print(r.line())
    core_pass = sum(1 for r in all_results if r.passed)
    print(f"\n{core_pass}/{len(all_results)} checks passed ({core_pass / max(1, len(all_results)):.1%})\n")

    regression_results: list[Result] = []
    if args.regression_checks.exists():
        with open(args.regression_checks, encoding="utf-8") as f:
            regression_checks = json.load(f)
        regression_results = check_regressions(document, regression_checks)
        print(f"=== Bug regression checklist ({len(regression_results)} known fixed bugs) ===")
        for r in regression_results:
            print(r.line())
        reg_pass = sum(1 for r in regression_results if r.passed)
        print(f"\n{reg_pass}/{len(regression_results)} regression checks passed\n")
        all_results += regression_results

    review_stats = None
    if args.review_csv:
        if not args.review_csv.exists():
            print(f"--review-csv {args.review_csv} not found — run pipeline.sample_review first")
        else:
            print("=== Human-review node sample ===")
            review_results, review_stats = evaluate_review_csv(args.review_csv)
            for r in review_results:
                print(r.line())
            print()

    overall_pass_count = sum(1 for r in all_results if r.passed)
    overall_ok = overall_pass_count == len(all_results) and (
        review_stats is None or (review_stats["incorrect"] == 0 and review_stats["unjudged"] == 0)
    )
    if review_stats:
        print(f"review sample: {review_stats['judged']}/{review_stats['total_rows']} judged, {review_stats['correct']} correct, {review_stats['incorrect']} incorrect")
    print("RESULT:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
