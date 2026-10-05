from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPECIMEN_GROUND_TRUTH = {
    "rancangan_kontrak": ("Rancangan Kontrak", "rancangan_kontrak1", "regression_checks.json"),
    "kontrakJasa": ("kontrakJasa", "kontrakJasa", "regression_checks/kontrakJasa.json"),
    "pembangunanRumah": ("pembangunanRumah", "pembangunanRumah", "regression_checks/pembangunanRumah.json"),
    "pembangunanSayap": ("pembangunanSayap", "pembangunanSayap", "regression_checks/pembangunanSayap.json"),
    "polres": ("polres", "polres", "regression_checks/polres.json"),
    "rehabGedung": ("rehabGedung", "rehabGedung", "regression_checks/rehabGedung.json"),
}
KNOWN_FAILING = {"pembangunanRumah": 3, "kontrakJasa": 2}
CONFIDENT_WRONG_BASELINE = {"pembangunanRumah": 2, "kontrakJasa": 1}


def _run(label: str, args: list[str], cwd: Path = PROJECT_DIR, announce: bool = True) -> tuple[bool, str, float]:
    started = time.time()
    proc = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok, elapsed = proc.returncode == 0, time.time() - started
    if announce:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({elapsed:.0f}s)")
    return ok, (proc.stdout or "") + (proc.stderr or ""), elapsed


def check_interpreter() -> bool:
    try:
        import fitz

        ok = "PyMuPDF" in (fitz.__doc__ or "")
    except Exception as exc:
        print(f"[FAIL] interpreter: importing fitz failed ({exc})")
        return False
    print(f"[{'PASS' if ok else 'FAIL'}] interpreter: {sys.executable}")
    if not ok:
        print("       `fitz` is not PyMuPDF — use JSONextraction/venv/Scripts/python.exe")
    return ok


RAW_DIR = PROJECT_DIR / "output" / "raw"
CODE_GLOBS = ("pipeline/*.py", "vocabulary/**/*", "profiles/**/*")
NON_EXTRACTION_MODULES = {"gate.py", "evaluate.py", "snapshot.py", "structure_score.py",
                          "synthetic_score.py", "sample_review.py"}


def _specimen_pdfs() -> list[Path]:
    return [PROJECT_DIR / "pdfs" / f"{pdf_stem}.pdf" for pdf_stem, _, _ in SPECIMEN_GROUND_TRUTH.values()]


def regenerate_raw() -> bool:
    ok, out, _ = _run("regenerate output/raw", ["-m", "pipeline.main", *map(str, _specimen_pdfs()),
                                                "--out", str(RAW_DIR)])
    if not ok:
        print("\n".join(f"       {line}" for line in out.splitlines() if "error" in line.lower() or "validation:" in line))
    return ok


def stale_raw_reasons() -> list[str]:
    from .main import DEFAULT_TREE_ENGINE

    code_mtime = max(
        (p.stat().st_mtime for pattern in CODE_GLOBS for p in PROJECT_DIR.glob(pattern)
         if p.is_file() and p.name not in NON_EXTRACTION_MODULES and "__pycache__" not in p.parts),
        default=0.0,
    )
    reasons = []
    for pdf in _specimen_pdfs():
        raw = RAW_DIR / f"{pdf.stem}_raw.json"
        if not raw.exists():
            reasons.append(f"{raw.name}: missing")
            continue
        engine = json.loads(raw.read_text(encoding="utf-8")).get("source", {}).get("tree_engine")
        if engine != DEFAULT_TREE_ENGINE:
            reasons.append(f"{raw.name}: tree_engine={engine}, default is {DEFAULT_TREE_ENGINE}")
        elif raw.stat().st_mtime < code_mtime:
            reasons.append(f"{raw.name}: older than the pipeline code")
    return reasons


def check_raw_fresh() -> bool:
    reasons = stale_raw_reasons()
    print(f"[{'FAIL' if reasons else 'PASS'}] output/raw fresh" + ("  (run the gate without --fast to regenerate)" if reasons else ""))
    for reason in reasons:
        print(f"       stale: {reason}")
    return not reasons


def evaluate_specimen(name: str, verbose: bool) -> bool:
    pdf_stem, gt_stem, checks = SPECIMEN_GROUND_TRUTH[name]
    ok_all = True
    for extra_label, extra in (("", []), (" [--no-label-locators]", ["--no-label-locators"])):
        args = [
            "-m", "pipeline.evaluate", f"output/raw/{pdf_stem}_raw.json",
            "--ground-truth", f"ground_truth/{gt_stem}.ground_truth.json",
            "--regression-checks", f"ground_truth/{checks}", *extra,
        ]
        label = f"evaluate {name}{extra_label}"
        ok, out, elapsed = _run(label, args, announce=False)
        note = ""
        confident_wrong = [line for line in out.splitlines() if "[CONFIDENT-WRONG]" in line]
        allowed = CONFIDENT_WRONG_BASELINE.get(name, 0)
        if len(confident_wrong) > allowed:
            ok = False
            note = f"  [{len(confident_wrong)} confidently wrong, baseline {allowed}]"
            for line in confident_wrong:
                print(f"       {line.strip()}")
        elif confident_wrong:
            note = f"  [{len(confident_wrong)} confidently wrong, at baseline {allowed}]"
        if not ok and name in KNOWN_FAILING:
            failures = out.count("[FAIL]") + out.count("[NOT_FOUND]") + out.count("[AMBIGUOUS]")
            ok = failures <= KNOWN_FAILING[name]
            note = (f"  [{failures} documented failures, baseline {KNOWN_FAILING[name]}]" if ok
                    else f"  [{failures} failures, ABOVE baseline {KNOWN_FAILING[name]}]")
        print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({elapsed:.0f}s){note}")
        ok_all = ok_all and ok
        if not ok and verbose:
            print("\n".join(f"       {line}" for line in out.splitlines() if "[FAIL]" in line or "RESULT" in line))
    return ok_all


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the full regression gate")
    parser.add_argument("--fast", action="store_true", help="Unit tests and evaluators only (no re-extraction)")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    started = time.time()
    results: list[tuple[str, bool]] = [("interpreter", check_interpreter())]

    if args.fast:
        results.append(("output/raw fresh", check_raw_fresh()))
    else:
        results.append(("regenerate output/raw", regenerate_raw()))

    for label, target in (("unit tests (pipeline)", "pipeline/tests"), ("unit tests (retrieval)", "retrieval/tests")):
        ok, _, _ = _run(label, ["-m", "unittest", "discover", "-s", target, "-t", "."])
        results.append((label, ok))

    for name in SPECIMEN_GROUND_TRUTH:
        results.append((f"evaluate {name}", evaluate_specimen(name, args.verbose)))

    if not args.fast:
        ok, out, _ = _run("snapshot diff", ["-m", "pipeline.snapshot", "diff"])
        results.append(("snapshot diff", ok))
        if not ok:
            print("\n".join(f"       {line}" for line in out.splitlines() if line.startswith("[DIFF]")))
            print("       Intended? re-record: python -m pipeline.snapshot record --reason \"...\"")

        ok, _, _ = _run("structure score", ["-m", "pipeline.structure_score", "--from-raw", "output/raw",
                                            "--only", *SPECIMEN_GROUND_TRUTH])
        results.append(("structure score", ok))

        ok, out, _ = _run("profile ablation", ["-m", "pipeline.structure_score", "--ablation"])
        results.append(("profile ablation", ok))
        if not ok and args.verbose:
            print("\n".join(f"       {line}" for line in out.splitlines() if line.startswith("[BELOW")))

        for engine in ("legacy", "relative"):
            ok, out, _ = _run(f"synthetic specimens [{engine}]",
                              ["-m", "pipeline.synthetic_score", "--tree-engine", engine])
            results.append((f"synthetic specimens [{engine}]", ok))
            if not ok:
                print("\n".join(f"       {line}" for line in out.splitlines() if line.startswith("[BELOW")))

        if not args.skip_retrieval:
            ok, out, _ = _run("retrieval gate", ["-m", "retrieval.retrieval_evaluate"])
            if "no store" in out or "no such collection" in out or "empty" in out:
                print("       skipped: no Chroma collection loaded")
                ok = True
            elif "EMBEDDING_MODEL is not set" in out:
                print("       skipped: retrieval not configured (no retrieval/.env)")
                ok = True
            results.append(("retrieval gate", ok))

    failed = [label for label, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed in {time.time() - started:.0f}s")
    if failed:
        print("failed: " + ", ".join(failed))
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
