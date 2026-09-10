"""Builds `<pdf-stem>_cleaned.json` — the reduced, keyword-only representation.

    python -m keywords.clean_json output/raw/polres_raw.json --out output/clean
    # -> output/clean/polres_cleaned.json        (--method rake, the default)
    # -> output/clean/polres_cleaned_yake.json   (--method yake)

The raw file stays untouched as the full-fidelity audit artifact; this
is a derived sibling. That separation is the whole point: the raw file remains
available for debugging and ground-truth evaluation, while the cleaned file is
what downstream storage and search consume at scale.

Every value-object in `core` ({value, raw, confidence, method, evidence,
candidates, flags}) is flattened to its `.value` here. The provenance those
wrappers carry is not lost — it is still in `raw_extraction.json`, which this
file names in `source.raw_extraction`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extractor import build_body, load_stopwords


def _value(core: dict, field: str):
    entry = core.get(field) or {}
    return entry.get("value") if isinstance(entry, dict) else None


def _confidence(core: dict, field: str):
    entry = core.get(field) or {}
    return entry.get("confidence") if isinstance(entry, dict) else None


def flatten_parties(parties) -> list[dict]:
    out = []
    for party in parties or []:
        if not isinstance(party, dict):
            continue
        out.append(
            {
                "role": party.get("role_label") or party.get("role"),
                "organization": (party.get("organization") or {}).get("value"),
                "representative": (party.get("representative") or {}).get("name"),
                "position": (party.get("representative") or {}).get("position"),
            }
        )
    return out


def flatten_dates(dates) -> list[dict]:
    return [
        {"type": d.get("type"), "date": d.get("date"), "precision": d.get("precision")}
        for d in dates or []
        if isinstance(d, dict)
    ]


def flatten_numbers(numbers) -> list[dict]:
    out = []
    for n in numbers or []:
        if not isinstance(n, dict):
            continue
        entry = {"type": n.get("type"), "subtype": n.get("subtype")}
        for key in ("amount", "unit", "value", "currency"):
            if n.get(key) is not None:
                entry[key] = n[key]
        out.append(entry)
    return out


def build_clean(document: dict, raw_path: Path, top_n: int = 40, keyword_method: str = "rake") -> dict:
    core = document.get("core") or {}
    status = core.get("_status") or {}
    source = document.get("source") or {}
    ocr = source.get("ocr")

    body = build_body(document, top_n=top_n, stopwords=load_stopwords(), method=keyword_method)

    clean = {
        "file_name": source.get("file"),
        "extraction_method": "ocr" if ocr else "native",
        "keyword_method": keyword_method,
        "document_type": _value(core, "document_type"),
        "contract_name": _value(core, "contract_name"),
        "contract_number": _value(core, "contract_number"),
        "parties": flatten_parties(_value(core, "parties")),
        "key_dates": flatten_dates(_value(core, "key_dates")),
        "key_numbers": flatten_numbers(_value(core, "key_numbers")),
        "body": body,
        "quality": {
            "overall_confidence": status.get("overall_confidence"),
            "fields_populated": status.get("fields_populated"),
            "requires_human_review": status.get("requires_human_review"),
            "review_reasons": status.get("review_reasons") or [],
            "pipeline_status": (document.get("quality") or {}).get("pipeline_status"),
            # Carried forward so a consumer of the cleaned file can tell how far
            # to trust it without opening the raw one. Absent on native output.
            "ocr_mean_confidence": (ocr or {}).get("mean_confidence"),
            "ocr_low_confidence_pages": (ocr or {}).get("low_confidence_pages"),
        },
        "source": {
            "raw_extraction": raw_path.name,
            "sha256": source.get("sha256"),
            "page_count": source.get("page_count"),
            "node_count": len(document.get("structure") or []),
            "table_count": len(document.get("tables") or []),
            "profile_id": (document.get("profile") or {}).get("profile_id"),
            "extracted_at": source.get("extracted_at"),
        },
    }
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reduce a <pdf-stem>_raw.json into a keyword-only <pdf-stem>_cleaned.json"
    )
    parser.add_argument("raw_path", type=Path, help="Path to the <pdf-stem>_raw.json file")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: alongside the input)")
    parser.add_argument("--top-n", type=int, default=40, help="Maximum mined keywords in `body` (default: 40)")
    parser.add_argument(
        "--method", choices=["yake", "rake"], default="rake",
        help="Statistical keyword extraction backend (default: rake)",
    )
    args = parser.parse_args()

    if not args.raw_path.exists():
        print(f"error: {args.raw_path} does not exist", file=sys.stderr)
        return 1

    with open(args.raw_path, encoding="utf-8") as f:
        document = json.load(f)

    clean = build_clean(document, args.raw_path, top_n=args.top_n, keyword_method=args.method)

    out_dir = args.out or args.raw_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    suffix = "_cleaned" if args.method == "rake" else f"_cleaned_{args.method}"
    out_path = out_dir / f"{stem}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    raw_size = args.raw_path.stat().st_size
    clean_size = out_path.stat().st_size
    print(f"document: {clean['contract_name']}")
    print(f"method:   {clean['extraction_method']} extraction, {clean['keyword_method']} keywords, profile={clean['source']['profile_id']}")
    print(f"body:     {len(clean['body'])} keywords")
    print(f"size:     {raw_size:,} -> {clean_size:,} bytes ({100 * clean_size / raw_size:.1f}% of raw)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
