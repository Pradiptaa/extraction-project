"""Generates a stratified human-review sample from raw_extraction.json.

Implements the sampling half of I.6 (analisis_pipeline_kontrak.md): pick a
fraction of nodes, stratified by sub_document so every part of the document
gets covered (not just whichever section happens to have the most nodes),
and dump them to a CSV a human fills in by comparing each row's text_raw
against the source PDF page.

Usage:
    python -m pipeline.sample_review output/raw_extraction.json \
        --out review/sample_for_review.csv --fraction 0.10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def build_sample(nodes: list[dict], fraction: float, seed: int, min_per_group: int) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        group = n.get("sub_document") or "unassigned"
        by_group[group].append(n)

    rng = random.Random(seed)
    sample: list[dict] = []
    for group, group_nodes in sorted(by_group.items()):
        k = max(min_per_group, round(len(group_nodes) * fraction))
        k = min(k, len(group_nodes))
        sample.extend(rng.sample(group_nodes, k))
    sample.sort(key=lambda n: (n.get("pages") or [0])[0], reverse=False)
    return sample


def write_csv(sample: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["node_id", "sub_document", "node_type", "label", "title", "pages", "reading_order", "text_raw", "correct_yn", "corrected_text", "notes"]
        )
        for n in sample:
            writer.writerow(
                [
                    n["node_id"],
                    n.get("sub_document") or "",
                    n["node_type"],
                    n.get("label") or "",
                    (n.get("title") or "").replace("\n", " "),
                    ";".join(str(p) for p in n.get("pages", [])),
                    n.get("reading_order", ""),
                    (n.get("text_raw") or "").replace("\n", " "),
                    "",   # correct_yn — human fills Y or N
                    "",   # corrected_text — human fills only if correct_yn = N
                    "",   # notes
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stratified human-review sample CSV from raw_extraction.json")
    parser.add_argument("raw_extraction", type=Path)
    parser.add_argument("--out", type=Path, default=Path("review/sample_for_review.csv"))
    parser.add_argument("--fraction", type=float, default=0.10, help="Sample fraction per sub_document (default 0.10)")
    parser.add_argument("--min-per-group", type=int, default=3, help="Minimum sampled nodes per sub_document group")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    args = parser.parse_args()

    with open(args.raw_extraction, encoding="utf-8") as f:
        document = json.load(f)

    nodes = document["structure"]
    sample = build_sample(nodes, args.fraction, args.seed, args.min_per_group)
    write_csv(sample, args.out)

    by_group = defaultdict(int)
    for n in sample:
        by_group[n.get("sub_document") or "unassigned"] += 1

    print(f"sampled {len(sample)} / {len(nodes)} nodes ({len(sample) / max(1, len(nodes)):.1%})")
    for group, count in sorted(by_group.items()):
        print(f"  {group}: {count}")
    print(f"wrote {args.out}")
    print("Fill in correct_yn (Y/N) for each row against the source PDF; put a fix in corrected_text when N.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
