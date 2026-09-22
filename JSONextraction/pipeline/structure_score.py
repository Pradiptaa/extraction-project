"""Label-independent structural scoring of a tree against a reference tree.

Two numbers, both blind to `node_type`, `sub_document` and titles:

* **boundary F1** — did the text get cut into the same units? Units are compared
  by their text alone, so a unit that keeps its content but is renamed, moved or
  re-typed still counts as found.
* **parent accuracy** — of the units found in both trees, how many sit under the
  same parent (by the parent's path, not its id, which is positional).

The reference is normally a recorded snapshot, which makes this the metric for
"the tree did not change shape", usable while labels are deliberately changing —
the Phase 3 engine swap in md/fix_plan.md.

Usage:
    python -m pipeline.structure_score                      # every specimen vs its snapshot
    python -m pipeline.structure_score --only rehabGedung --min-boundary-f1 0.99
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .snapshot import DEFAULT_SNAPSHOT_DIR, current_snapshots, load_manifest


@dataclass
class StructureScore:
    specimen: str
    reference_nodes: int
    candidate_nodes: int
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    parent_accuracy: float
    depth_accuracy: float
    matched: int

    def line(self) -> str:
        return (
            f"{self.specimen:20} boundary_f1={self.boundary_f1:.4f} "
            f"(p={self.boundary_precision:.4f} r={self.boundary_recall:.4f}) "
            f"parent_acc={self.parent_accuracy:.4f} depth_acc={self.depth_accuracy:.4f} "
            f"nodes={self.candidate_nodes}/{self.reference_nodes} matched={self.matched}"
        )


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score(reference: list[dict], candidate: list[dict]) -> tuple[float, float, float, float, float, int]:
    """(precision, recall, f1, parent_accuracy, depth_accuracy, matched)."""
    # Text alone is the unit's identity: an empty-text node (a bare heading) is
    # identified by its title instead, so those are not all one unit.
    unit = lambda r: r["text_sha"] if r["text_len"] else f"title:{r['title']!r}"  # noqa: E731
    ref_units, cand_units = Counter(map(unit, reference)), Counter(map(unit, candidate))
    overlap = sum((ref_units & cand_units).values())
    precision = overlap / max(1, sum(cand_units.values()))
    recall = overlap / max(1, sum(ref_units.values()))

    by_unit: dict[str, list[dict]] = {}
    for rec in candidate:
        by_unit.setdefault(unit(rec), []).append(rec)
    same_parent = same_depth = matched = 0
    for rec in reference:
        bucket = by_unit.get(unit(rec))
        if not bucket:
            continue
        other = bucket.pop(0)
        matched += 1
        same_parent += rec["parent_path"] == other["parent_path"]
        same_depth += rec["depth"] == other["depth"]
    return (
        precision, recall, _f1(precision, recall),
        same_parent / max(1, matched), same_depth / max(1, matched), matched,
    )


def score_snapshot(reference: dict, candidate: dict) -> StructureScore:
    p, r, f1, parent, depth, matched = score(reference["nodes"], candidate["nodes"])
    return StructureScore(
        specimen=candidate.get("specimen") or reference.get("specimen"),
        reference_nodes=len(reference["nodes"]), candidate_nodes=len(candidate["nodes"]),
        boundary_precision=p, boundary_recall=r, boundary_f1=f1,
        parent_accuracy=parent, depth_accuracy=depth, matched=matched,
    )


ABLATION_SUFFIX = "__generic"
ABLATION_FLOOR_NAME = "ablation_floor.json"


def run_ablation(snapshot_dir: Path, record: bool, tolerance: float) -> int:
    """How much of the tree survives when no profile matches.

    This is acceptance criterion 1 of the review — profiles should change
    *labels*, not *shape*. Today they change both, so the floor starts well
    below 1.0; Phase 3 is what should lift it. The floor may only rise, which is
    why a drop fails and an improvement asks to be recorded."""
    floor_path = snapshot_dir / ABLATION_FLOOR_NAME
    floors = json.loads(floor_path.read_text(encoding="utf-8")).get("floors", {}) if floor_path.exists() else {}

    pairs = [
        (s["name"], s["name"][: -len(ABLATION_SUFFIX)])
        for s in load_manifest(snapshot_dir) if s["name"].endswith(ABLATION_SUFFIX)
    ]
    if not pairs:
        print("no profile-ablation specimens in the manifest")
        return 1

    measured, failed, improved = {}, False, False
    for candidate_name, reference_name in pairs:
        candidate = snapshot_dir / f"{candidate_name}.snapshot.json"
        reference = snapshot_dir / f"{reference_name}.snapshot.json"
        if not (candidate.exists() and reference.exists()):
            print(f"[MISS ] {candidate_name}: record both snapshots first")
            failed = True
            continue
        s = score_snapshot(json.loads(reference.read_text(encoding="utf-8")),
                           json.loads(candidate.read_text(encoding="utf-8")))
        measured[reference_name] = {"boundary_f1": round(s.boundary_f1, 4),
                                    "parent_accuracy": round(s.parent_accuracy, 4)}
        floor = floors.get(reference_name)
        status = "OK"
        if floor:
            if s.boundary_f1 < floor["boundary_f1"] - tolerance or s.parent_accuracy < floor["parent_accuracy"] - tolerance:
                status, failed = "BELOW", True
            elif s.boundary_f1 > floor["boundary_f1"] + tolerance or s.parent_accuracy > floor["parent_accuracy"] + tolerance:
                status, improved = "BETTER", True
        else:
            status = "NEW"
        print(f"[{status:6}] {reference_name:20} boundary_f1={s.boundary_f1:.4f} parent_acc={s.parent_accuracy:.4f}"
              + (f"  floor={floor['boundary_f1']:.4f}/{floor['parent_accuracy']:.4f}" if floor else ""))

    if record:
        floor_path.write_text(json.dumps({
            "_purpose": "Profile-independence floor: scores of each specimen extracted with the fallback profile "
                        "only, against the same specimen extracted normally. Raise it when Phase 3 improves it; "
                        "never lower it to make the gate pass.",
            "floors": measured,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"recorded floor for {len(measured)} specimens")
        return 0
    if improved:
        print("Improved — re-record with: python -m pipeline.structure_score --ablation --record")
    print("RESULT:", "BELOW FLOOR" if failed else "PASS")
    return 1 if failed else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Score tree shape against recorded snapshots")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--from-raw", type=Path, default=None)
    parser.add_argument("--min-boundary-f1", type=float, default=1.0,
                        help="Fail below this (default 1.0: shape must be identical)")
    parser.add_argument("--min-parent-accuracy", type=float, default=1.0)
    parser.add_argument("--ablation", action="store_true",
                        help="Score every `<name>__generic` snapshot against `<name>`: what the pipeline "
                             "loses when no profile matches. Held at a recorded floor, which must rise.")
    parser.add_argument("--record", action="store_true", help="With --ablation: rewrite the floor file")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="With --ablation: allowed drop below the recorded floor (default 0.005)")
    parser.add_argument("--pair", nargs=2, metavar=("CANDIDATE", "REFERENCE"),
                        help="Score two recorded snapshots against each other instead of re-extracting, "
                             "e.g. --pair rehabGedung__generic rehabGedung")
    args = parser.parse_args()

    if args.ablation:
        return run_ablation(args.snapshot_dir, args.record, args.tolerance)

    if args.pair:
        candidate_name, reference_name = args.pair
        snaps = {}
        for name in (candidate_name, reference_name):
            path = args.snapshot_dir / f"{name}.snapshot.json"
            if not path.exists():
                print(f"no snapshot for {name!r}")
                return 1
            snaps[name] = json.loads(path.read_text(encoding="utf-8"))
        s = score_snapshot(snaps[reference_name], snaps[candidate_name])
        s.specimen = f"{candidate_name} vs {reference_name}"
        below = s.boundary_f1 < args.min_boundary_f1 or s.parent_accuracy < args.min_parent_accuracy
        print(f"[{'BELOW' if below else 'OK':5}] {s.line()}")
        return 1 if below else 0

    specs = [s for s in load_manifest(args.snapshot_dir) if not args.only or s["name"] in args.only]
    candidates = current_snapshots(specs, args.from_raw)
    failed = False
    for spec in specs:
        path = args.snapshot_dir / f"{spec['name']}.snapshot.json"
        if not path.exists():
            print(f"{spec['name']:20} NO SNAPSHOT — record one first")
            failed = True
            continue
        s = score_snapshot(json.loads(path.read_text(encoding="utf-8")), candidates[spec["name"]])
        below = s.boundary_f1 < args.min_boundary_f1 or s.parent_accuracy < args.min_parent_accuracy
        failed = failed or below
        print(f"[{'BELOW' if below else 'OK':5}] {s.line()}")
    print("RESULT:", "BELOW THRESHOLD" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
