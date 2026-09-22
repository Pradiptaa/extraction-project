"""Golden snapshots of extraction output, and a categorized diff against them.

A snapshot is a normalized projection of `<pdf-stem>_raw.json` that drops what
changes run to run (timestamps) or shifts with any upstream edit (char spans,
positional node_ids), so that a diff shows only what a code change did to the
extraction. It is the regression contract for refactors: a pure refactor must
diff empty, and any other change must show only the differences it intended.

Usage:
    python -m pipeline.snapshot record --reason "initial baseline"
    python -m pipeline.snapshot diff
    python -m pipeline.snapshot diff --from-raw output/raw      # skip re-extraction
    python -m pipeline.snapshot diff --only polres --verbose
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = PROJECT_DIR / "ground_truth" / "snapshots"
MANIFEST_NAME = "manifest.json"
CHANGELOG_NAME = "CHANGELOG.jsonl"
SNAPSHOT_FORMAT = "1"

# Diff categories. `structure` is what the tree looks like; `labels` is what the
# profile and classifiers call its nodes — kept apart because the planned tree
# redesign must change the second without the first.
CATEGORIES = ("meta", "pages", "structure", "text", "labels", "core", "tables", "entities", "quality")


def _sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def _strip_spans(value):
    """Removes char offsets, which move whenever any earlier text changes."""
    if isinstance(value, dict):
        return {k: _strip_spans(v) for k, v in value.items() if k not in ("char_span", "evidence")}
    if isinstance(value, list):
        return [_strip_spans(v) for v in value]
    return value


def _node_records(structure: list[dict]) -> list[dict]:
    by_id = {n["node_id"]: n for n in structure}
    records = []
    for n in sorted(structure, key=lambda n: n.get("reading_order", 0)):
        parent = by_id.get(n.get("parent_id"))
        records.append(
            {
                "path": "/".join(n.get("path") or []),
                "parent_path": "/".join(parent.get("path") or []) if parent else None,
                "depth": n.get("depth"),
                "node_type": n.get("node_type"),
                "numbering_style": n.get("numbering_style"),
                "label_normalized": n.get("label_normalized"),
                "sub_document": n.get("sub_document"),
                "pages": n.get("pages"),
                "title": n.get("title"),
                "text_sha": _sha(n.get("text_raw") or ""),
                "text_len": len(n.get("text_raw") or ""),
                "text_head": (n.get("text_raw") or "")[:80],
                "flags": sorted((n.get("extraction") or {}).get("flags") or []),
                "refs_out": sorted(r.get("raw", "") for r in n.get("refs_out") or []),
            }
        )
    return records


def normalize(raw: dict, specimen: str) -> dict:
    """The snapshot of one raw extraction file."""
    source = raw.get("source") or {}
    entities = raw.get("entities") or []
    return {
        "snapshot_format": SNAPSHOT_FORMAT,
        "specimen": specimen,
        "meta": {
            "file": source.get("file"),
            "sha256": source.get("sha256"),
            "page_count": source.get("page_count"),
            "schema_version": raw.get("schema_version"),
            "pipeline_version": source.get("pipeline_version"),
            "profile_id": (raw.get("profile") or {}).get("profile_id"),
            "profile_score": (raw.get("profile") or {}).get("match_score"),
        },
        "pages": [
            {
                "page": p.get("page"),
                "layout_type": p.get("layout_type"),
                "sub_document": p.get("sub_document"),
                "extraction_method": p.get("extraction_method"),
                "page_label": p.get("page_label"),
                "text_sha": _sha(p.get("raw_text") or ""),
                "char_count": p.get("char_count"),
            }
            for p in raw.get("pages") or []
        ],
        "nodes": _node_records(raw.get("structure") or []),
        "core": _strip_spans(raw.get("core") or {}),
        "tables": [
            {
                "table_id": t.get("table_id"),
                "headers": t.get("headers"),
                "row_count": len(t.get("rows") or []),
                "rows_sha": _sha(json.dumps([r.get("cells") for r in t.get("rows") or []], ensure_ascii=False)),
                "refs_out": sorted(ref.get("raw", "") for r in t.get("rows") or [] for ref in r.get("refs_out") or []),
            }
            for t in raw.get("tables") or []
        ],
        "entities": sorted(
            json.dumps(_strip_spans({k: v for k, v in e.items() if k != "node_id"}), ensure_ascii=False, sort_keys=True)
            for e in entities
        ),
        "quality": {
            "pipeline_status": (raw.get("quality") or {}).get("pipeline_status"),
            "checks": {c["check"]: f"{c['status']}: {c['detail']}" for c in (raw.get("quality") or {}).get("checks") or []},
            "tree_quality_flags": (raw.get("quality") or {}).get("tree_quality_flags") or [],
        },
    }


# --------------------------------------------------------------------------
# diffing
# --------------------------------------------------------------------------

_STRUCTURE_FIELDS = ("parent_path", "depth")
_TEXT_FIELDS = ("text_sha", "text_len", "pages")
_LABEL_FIELDS = ("node_type", "numbering_style", "label_normalized", "sub_document", "title", "flags", "refs_out")


@dataclass
class SpecimenDiff:
    specimen: str
    changes: dict[str, list[str]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})

    @property
    def empty(self) -> bool:
        return not any(self.changes.values())

    def add(self, category: str, line: str) -> None:
        self.changes[category].append(line)


def _node_label(rec: dict) -> str:
    where = rec["path"] or f"<{rec['node_type']}>"
    return f"{where} [{(rec['text_head'] or rec['title'] or '')[:40]!r}]"


# Unnumbered nodes all share path "", so path alone would pair unrelated
# headings; they must also read alike to count as the same node.
_UNNUMBERED_PAIR_MIN_SIMILARITY = 0.6


def _surface(rec: dict) -> str:
    return f"{rec['title'] or ''} {rec['text_head']}".strip().lower()


def _find_pair(rec: dict, candidates: list[dict]) -> dict | None:
    same_path = [n for n in candidates if n["path"] == rec["path"]]
    if rec["path"]:
        return same_path[0] if same_path else None
    scored = [(difflib.SequenceMatcher(None, _surface(rec), _surface(n)).ratio(), i, n) for i, n in enumerate(same_path)]
    best = max(scored, default=None, key=lambda t: (t[0], -t[1]))
    return best[2] if best and best[0] >= _UNNUMBERED_PAIR_MIN_SIMILARITY else None


def _diff_nodes(old: list[dict], new: list[dict], out: SpecimenDiff) -> None:
    """Aligns nodes by (path, text) in reading order, then pairs nodes inside
    changed runs, so edited text reads as a change, not a remove + add."""
    key = lambda r: (r["path"], r["text_sha"], r["title"])  # noqa: E731
    matcher = difflib.SequenceMatcher(None, [key(r) for r in old], [key(r) for r in new], autojunk=False)
    pairs: list[tuple[dict, dict]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(old[i1:i2], new[j1:j2]))
            continue
        remaining_new = list(new[j1:j2])
        for rec in old[i1:i2]:
            match = _find_pair(rec, remaining_new)
            if match is None:
                out.add("structure", f"- removed node {_node_label(rec)}")
            else:
                remaining_new.remove(match)
                pairs.append((rec, match))
        for rec in remaining_new:
            out.add("structure", f"+ added node {_node_label(rec)}")

    for a, b in pairs:
        for category, fields in (("structure", _STRUCTURE_FIELDS), ("text", _TEXT_FIELDS), ("labels", _LABEL_FIELDS)):
            for f in fields:
                if a[f] != b[f]:
                    if f == "text_sha":
                        out.add(category, f"~ {_node_label(a)} text: {a['text_head']!r} -> {b['text_head']!r}")
                    elif f != "text_len":
                        out.add(category, f"~ {_node_label(a)} {f}: {a[f]!r} -> {b[f]!r}")


def _diff_flat(category: str, old, new, out: SpecimenDiff, prefix: str = "") -> None:
    """Leaf-level diff of nested dicts/lists."""
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new), key=str):
            _diff_flat(category, old.get(k), new.get(k), out, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            _diff_flat(category, a, b, out, f"{prefix}[{i}]")
    elif old != new:
        out.add(category, f"~ {prefix}: {json.dumps(old, ensure_ascii=False)[:160]} -> {json.dumps(new, ensure_ascii=False)[:160]}")


def _diff_multiset(category: str, old: list[str], new: list[str], out: SpecimenDiff) -> None:
    from collections import Counter

    before, after = Counter(old), Counter(new)
    for item in sorted((before - after).elements()):
        out.add(category, f"- {item[:200]}")
    for item in sorted((after - before).elements()):
        out.add(category, f"+ {item[:200]}")


def diff_snapshots(old: dict, new: dict) -> SpecimenDiff:
    out = SpecimenDiff(new.get("specimen") or old.get("specimen"))
    _diff_flat("meta", old["meta"], new["meta"], out)
    _diff_flat("pages", old["pages"], new["pages"], out)
    if len(old["pages"]) != len(new["pages"]):
        out.add("pages", f"~ page count {len(old['pages'])} -> {len(new['pages'])}")
    _diff_nodes(old["nodes"], new["nodes"], out)
    _diff_flat("core", old["core"], new["core"], out)
    _diff_flat("tables", old["tables"], new["tables"], out)
    if len(old["tables"]) != len(new["tables"]):
        out.add("tables", f"~ table count {len(old['tables'])} -> {len(new['tables'])}")
    _diff_multiset("entities", old["entities"], new["entities"], out)
    _diff_flat("quality", old["quality"], new["quality"], out)
    return out


# --------------------------------------------------------------------------
# specimens and extraction
# --------------------------------------------------------------------------

def load_manifest(snapshot_dir: Path) -> list[dict]:
    """[{"name", "pdf", "engine": "native"|"ocr"}], relative to the project dir.

    Entries without a `name` are comments and are skipped."""
    specimens = json.loads((snapshot_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["specimens"]
    return [s for s in specimens if s.get("name")]


def generic_only_profile_dir(work_dir: Path) -> Path:
    """A profile directory holding nothing but the mandatory fallback.

    Simulates "no profile matched" on a document we have ground truth for, which
    is how the profile-ablation specimens measure what the profile is currently
    load-bearing for. Copied from the real profile at run time, so it can never
    drift from it."""
    from .profiles import DEFAULT_PROFILE_DIR, FALLBACK_PROFILE_ID

    from vocabulary import BASE_VOCABULARY_ID

    out = work_dir / "profiles_generic_only"
    out.mkdir(parents=True, exist_ok=True)
    # The fallback profile and the base vocabulary: ablation removes the
    # *profile*, not the language. Without base_id.json every label dictionary
    # would be empty, which measures a missing file rather than a missing profile.
    for name in (FALLBACK_PROFILE_ID, BASE_VOCABULARY_ID):
        source = DEFAULT_PROFILE_DIR / f"{name}.json"
        if source.exists():
            (out / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def extract(spec: dict, work_dir: Path) -> dict:
    """Runs the specimen's pipeline in-process and returns its raw document."""
    pdf = PROJECT_DIR / spec["pdf"]
    profile_dir = generic_only_profile_dir(work_dir) if spec.get("profile_mode") == "generic_only" else None
    if spec["engine"] == "native":
        from .main import run_pipeline

        return run_pipeline(pdf, work_dir, profile_dir)
    if spec["engine"] == "ocr":
        from . import ocr_main

        return ocr_main.run_ocr_pipeline(pdf, work_dir, profile_dir, **spec.get("ocr_args", {}))
    raise ValueError(f"unknown engine {spec['engine']!r} for {spec['name']}")


def current_snapshots(specs: list[dict], from_raw: Path | None) -> dict[str, dict]:
    snapshots = {}
    with tempfile.TemporaryDirectory() as tmp:
        for spec in specs:
            if from_raw is not None:
                path = from_raw / f"{spec.get('raw_stem', Path(spec['pdf']).stem)}_raw.json"
                raw = json.loads(path.read_text(encoding="utf-8"))
            else:
                print(f"  extracting {spec['name']} ({spec['engine']}) ...", file=sys.stderr)
                raw = extract(spec, Path(tmp))
            snap = normalize(raw, spec["name"])
            if spec["engine"] == "ocr":
                # Recognition output depends on the engine as much as on this
                # code, so an upgrade must be visible as an engine change.
                snap["meta"]["tesseract_version"] = tesseract_version()
            snapshots[spec["name"]] = snap
    return snapshots


def tesseract_version() -> str:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception as exc:  # pragma: no cover - environment probe
        return f"unavailable: {exc}"


def _snapshot_path(snapshot_dir: Path, name: str) -> Path:
    return snapshot_dir / f"{name}.snapshot.json"


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_diff(d: SpecimenDiff, verbose: bool, limit: int = 15) -> None:
    if d.empty:
        print(f"[SAME] {d.specimen}")
        return
    summary = ", ".join(f"{c}={len(v)}" for c, v in d.changes.items() if v)
    print(f"[DIFF] {d.specimen}: {summary}")
    for category, lines in d.changes.items():
        if not lines:
            continue
        print(f"  {category}:")
        shown = lines if verbose else lines[:limit]
        for line in shown:
            print(f"    {line}")
        if len(lines) > len(shown):
            print(f"    ... {len(lines) - len(shown)} more (--verbose)")


def cmd_diff(args) -> int:
    specs = [s for s in load_manifest(args.snapshot_dir) if not args.only or s["name"] in args.only]
    current = current_snapshots(specs, args.from_raw)
    any_diff = False
    for spec in specs:
        path = _snapshot_path(args.snapshot_dir, spec["name"])
        if not path.exists():
            print(f"[MISSING] {spec['name']}: no recorded snapshot at {path.name}")
            any_diff = True
            continue
        d = diff_snapshots(json.loads(path.read_text(encoding="utf-8")), current[spec["name"]])
        _print_diff(d, args.verbose)
        any_diff = any_diff or not d.empty
    print("RESULT:", "DIFF — every change must be intended, then re-recorded with a reason" if any_diff else "no differences")
    return 1 if any_diff else 0


def cmd_record(args) -> int:
    specs = [s for s in load_manifest(args.snapshot_dir) if not args.only or s["name"] in args.only]
    current = current_snapshots(specs, args.from_raw)
    log_entries = []
    for spec in specs:
        path = _snapshot_path(args.snapshot_dir, spec["name"])
        if path.exists():
            d = diff_snapshots(json.loads(path.read_text(encoding="utf-8")), current[spec["name"]])
            if d.empty:
                print(f"[SAME] {spec['name']}: unchanged, not rewritten")
                continue
            summary = {c: len(v) for c, v in d.changes.items() if v}
        else:
            summary = {"new": 1}
        _write(path, current[spec["name"]])
        print(f"[WROTE] {spec['name']}: {summary}")
        log_entries.append({"specimen": spec["name"], "changes": summary})

    if log_entries:
        entry = {
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": args.reason,
            "specimens": log_entries,
        }
        with open(args.snapshot_dir / CHANGELOG_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    # Windows consoles default to a legacy code page that can't print "–" or "‰".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Record or diff golden extraction snapshots")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("diff", "record"):
        p = sub.add_parser(name)
        p.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
        p.add_argument("--only", nargs="+", help="Specimen names from the manifest")
        p.add_argument("--from-raw", type=Path, default=None,
                       help="Read <stem>_raw.json from this directory instead of re-extracting")
        if name == "diff":
            p.add_argument("--verbose", action="store_true")
        else:
            p.add_argument("--reason", required=True, help="Why the snapshot changed; appended to CHANGELOG.jsonl")
    args = parser.parse_args()
    return cmd_diff(args) if args.command == "diff" else cmd_record(args)


if __name__ == "__main__":
    raise SystemExit(main())
