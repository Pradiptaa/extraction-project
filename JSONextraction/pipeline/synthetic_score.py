"""Score extraction against an *authored* tree (the synthetic specimens).

Unlike `structure_score`, which compares a run to a previous run, this compares
a run to structure that is known to be correct because the same tree rendered
the PDF. That makes it the only measure here of "is the extraction right?"
rather than "did the extraction change?".

Three numbers per specimen:

* **unit recall** — of the authored units, how many appear at all (matched by
  their text, so a unit found under a wrong name still counts)
* **path accuracy** — of those, how many carry the authored path
* **parent accuracy** — of those, how many sit under the authored parent

All three ignore `node_type` and `sub_document`, so they stay meaningful while
Phase 3 changes what things are called.

    python -m pipeline.synthetic_score                 # extract and score them all
    python -m pipeline.synthetic_score --only inverted_nesting --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .main import DEFAULT_TREE_ENGINE

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRUTH_DIR = PROJECT_DIR / "ground_truth" / "synthetic"
BASELINE_PATH = TRUTH_DIR / "score_baseline.json"
# OCR reads the same words through an imperfect channel, so text matching is
# fuzzy there; the native path is expected to reproduce text exactly.
OCR_SIMILARITY = 0.75


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _similar(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass
class UnitResult:
    path: list[str]
    found: bool
    path_ok: bool
    parent_ok: bool
    detail: str = ""


@dataclass
class SpecimenScore:
    name: str
    results: list[UnitResult] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return sum(r.found for r in self.results) / max(1, len(self.results))

    @property
    def path_accuracy(self) -> float:
        found = [r for r in self.results if r.found]
        return sum(r.path_ok for r in found) / max(1, len(found))

    @property
    def parent_accuracy(self) -> float:
        found = [r for r in self.results if r.found]
        return sum(r.parent_ok for r in found) / max(1, len(found))

    def line(self) -> str:
        return (f"{self.name:24} unit_recall={self.recall:.3f} path_acc={self.path_accuracy:.3f} "
                f"parent_acc={self.parent_accuracy:.3f} units={len(self.results)}")


def _node_surface(node: dict) -> str:
    return _norm(f"{node.get('title') or ''} {node.get('text_raw') or ''}")


def score_document(authored: dict, document: dict) -> SpecimenScore:
    """A unit is found when some node's text carries the authored text. Title
    and body are checked together, because which of the two a heading's words
    land in is a labelling decision this metric deliberately ignores."""
    nodes = document.get("structure") or []
    by_id = {n["node_id"]: n for n in nodes}
    surfaces = [(n, _node_surface(n)) for n in nodes]
    fuzzy = authored.get("engine") == "ocr"
    score = SpecimenScore(authored["name"])

    for unit in authored["units"]:
        wanted = _norm(f"{unit['title']} {unit['body']}")
        if not wanted:
            continue
        # Best candidate, not the first: a short unit ("PELAKSANAAN") also
        # appears inside unrelated prose ("Masa Pelaksanaan ditentukan..."), and
        # taking the first containing node scored the wrong one.
        wanted_label = unit["path"][-1] if unit["path"] else None
        best: tuple[int, int, dict] | None = None
        for order, (node, surface) in enumerate(surfaces):
            if not surface:
                continue
            if wanted in surface:
                rank = 3 if surface.startswith(wanted) else 1
            elif fuzzy and _similar(wanted, surface) >= OCR_SIMILARITY:
                rank = 2
            else:
                continue
            if wanted_label and str(node.get("label_normalized") or "").lower() == wanted_label.lower():
                rank += 3          # the unit's own label settles it
            if best is None or rank > best[0]:
                best = (rank, order, node)
        match, detail = (best[2] if best else None), ""
        if match is None:
            # A unit split across nodes still counts as found if some node holds
            # its opening — a split is a path error, not a lost unit.
            head = " ".join(wanted.split()[:8])
            for node, surface in surfaces:
                if head and head in surface:
                    match, detail = node, "partial: text split across nodes"
                    break
        if match is None:
            score.results.append(UnitResult(unit["path"], False, False, False, f"not found: {wanted[:60]!r}"))
            continue
        parent = by_id.get(match.get("parent_id"))
        parent_path = (parent.get("path") or []) if parent else []
        path_ok = (match.get("path") or []) == unit["path"]
        parent_ok = parent_path == unit["parent_path"]
        if not path_ok:
            detail = (detail + f" path={match.get('path')} want={unit['path']}").strip()
        elif not parent_ok:
            detail = (detail + f" parent={parent_path} want={unit['parent_path']}").strip()
        score.results.append(UnitResult(unit["path"], True, path_ok, parent_ok, detail))
    return score


def extract(authored: dict, work_dir: Path, tree_engine: str) -> dict:
    pdf = PROJECT_DIR / authored["pdf"]
    if authored.get("engine") == "ocr":
        from . import ocr_main

        return ocr_main.run_ocr_pipeline(pdf, work_dir, tree_engine=tree_engine)
    from .main import run_pipeline

    return run_pipeline(pdf, work_dir, None, tree_engine)


def load_authored(only: list[str] | None) -> list[dict]:
    out = []
    for path in sorted(TRUTH_DIR.glob("*.authored.json")):
        authored = json.loads(path.read_text(encoding="utf-8"))
        if not only or authored["name"] in only:
            out.append(authored)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Score extraction against the authored synthetic specimens")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--verbose", action="store_true", help="List every unit that is missing or misplaced")
    parser.add_argument("--record", action="store_true", help="Record the current scores as the floor")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--tree-engine", choices=("legacy", "relative"), default=DEFAULT_TREE_ENGINE,
                        help="Which depth engine to score (default: the pipeline's own default)")
    args = parser.parse_args()

    authored_specs = load_authored(args.only)
    if not authored_specs:
        print("no authored specimens — run `python -m synthetic.make_specimens` first")
        return 1

    # Floors are per engine: the two build different trees on purpose, so one
    # set of numbers cannot judge both.
    stored = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else {}
    all_floors = stored.get("floors_by_engine", {})
    baseline = all_floors.get(args.tree_engine, {})
    measured, failed, improved = {}, False, False

    with tempfile.TemporaryDirectory() as tmp:
        for authored in authored_specs:
            document = extract(authored, Path(tmp), args.tree_engine)
            score = score_document(authored, document)
            measured[score.name] = {"unit_recall": round(score.recall, 3),
                                    "path_accuracy": round(score.path_accuracy, 3),
                                    "parent_accuracy": round(score.parent_accuracy, 3)}
            floor, status = baseline.get(score.name), "NEW"
            if floor:
                worse = any(measured[score.name][k] < floor[k] - args.tolerance for k in floor)
                better = any(measured[score.name][k] > floor[k] + args.tolerance for k in floor)
                status = "BELOW" if worse else ("BETTER" if better else "OK")
                failed = failed or worse
                improved = improved or better
            print(f"[{status:6}] {args.tree_engine:8} {score.line()}")
            if args.verbose:
                for r in score.results:
                    if not (r.found and r.path_ok and r.parent_ok):
                        print(f"         {'/'.join(r.path):20} {r.detail}")

    if args.record:
        all_floors[args.tree_engine] = measured
        BASELINE_PATH.write_text(json.dumps({
            "_purpose": "Scores of the current extractor against the authored synthetic specimens. These are "
                        "correctness floors on documents outside the Perpres-16 family: they should rise as "
                        "the generalization phases land, and must never fall.",
            "floors": measured,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"recorded floor for {len(measured)} specimens")
        return 0
    if improved:
        print("Improved — re-record with: python -m pipeline.synthetic_score --record")
    print("RESULT:", "BELOW FLOOR" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
