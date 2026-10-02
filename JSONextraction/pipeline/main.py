from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

import vocabulary

from . import core_fields, entities as entities_mod, profiles as profiles_mod
from .blocks import extract_table_blocks, extract_text_blocks
from .layout import classify_layout
from .probe import probe_document
from .router import route_pages
from . import field_context, segments
from .schema import SCHEMA_VERSION
from .tree import build_tree
from .validate import run_validation

DEFAULT_TREE_ENGINE = os.environ.get("TREE_ENGINE", "relative")

PAGE_LABEL_RE = re.compile(r"(?:^|\n)\s*-?\s*(\d{1,4})\s*-?\s*$")
PLACEHOLDER_COUNT_RE = re.compile(r"…|\.{4,}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_page_label(raw_text: str) -> str | None:
    tail = "\n".join(raw_text.strip().splitlines()[-2:])
    m = PAGE_LABEL_RE.search(tail)
    return m.group(1) if m else None


def guess_document_status(full_text: str) -> str:
    placeholder_hits = len(PLACEHOLDER_COUNT_RE.findall(full_text))
    return "draft_template" if placeholder_hits >= 5 else "executed_or_unclassified"


def prelim_page_text(blocks: list) -> str:
    return "\n".join(b.text for b in blocks)


def assign_sub_documents(page_order: list[int], page_raw_text: dict[int, str], profile: dict) -> dict[int, str | None]:
    markers = profile.get("sub_document_markers", [])
    first_seen_page: dict[str, int] = {}
    for page in page_order:
        text = page_raw_text.get(page, "")
        for marker in markers:
            name = marker["name"]
            if name in first_seen_page:
                continue
            if re.search(marker["start"], text, re.MULTILINE):
                first_seen_page[name] = page

    events = sorted(first_seen_page.items(), key=lambda kv: kv[1])
    result: dict[int, str | None] = {}
    current: str | None = None
    event_idx = 0
    for page in page_order:
        while event_idx < len(events) and events[event_idx][1] <= page:
            current = events[event_idx][0]
            event_idx += 1
        result[page] = current
    return result


CLAUSE_REF_COLUMN_RE = re.compile(r"\b(pasal|ssuk|sskk|klausul|ketentuan|ref)\b", re.IGNORECASE)
DOTTED_REF_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){1,2}\b")
ANY_REF_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){0,2}\b")


def build_table_entries(table_blocks_by_page: dict[int, list], label_index: dict[str, str]) -> list[dict]:
    tables = []
    for page, tblocks in sorted(table_blocks_by_page.items()):
        for t_idx, t in enumerate(tblocks):
            rows_out = []
            headers = t.rows[0] if t.rows else []
            first_column = [(row[0] if row else "") or "" for row in t.rows]
            keyed_column = (
                CLAUSE_REF_COLUMN_RE.search((headers[0] if headers else "") or "")
                or any(DOTTED_REF_RE.search(cell) for cell in first_column)
            )
            clause_ref_re = ANY_REF_RE if keyed_column else DOTTED_REF_RE
            for row in t.rows[1:] if len(t.rows) > 1 else t.rows:
                first_cell = row[0] if row else ""
                refs = []
                for m in clause_ref_re.finditer(first_cell or ""):
                    node_id = label_index.get(m.group(0))
                    refs.append({"raw": m.group(0), "resolved_node_id": node_id, "type": "internal"})
                rows_out.append({"cells": row, "refs_out": refs})
            tables.append(
                {
                    "table_id": f"t_{page:03d}_{t_idx}",
                    "page": page,
                    "headers": headers,
                    "rows": rows_out,
                    "bbox": t.bbox,
                    "extraction_method": t.extraction_method,
                }
            )
    return tables


def run_pipeline(pdf_path: Path, output_dir: Path, profile_dir: Path | None = None,
                 tree_engine: str = DEFAULT_TREE_ENGINE) -> dict:
    probes = probe_document(str(pdf_path))
    route_decisions = route_pages(probes)
    route_by_page = {r.page: r for r in route_decisions}

    detect_parallel = tree_engine == "relative"
    layouts = {p.page: classify_layout(p, detect_parallel) for p in probes}
    layout_type_by_page = {page: info.layout_type for page, info in layouts.items()}

    ruled_pages = [p for p, lt in layout_type_by_page.items() if lt == "ruled_table"]
    table_blocks_by_page = extract_table_blocks(str(pdf_path), ruled_pages)

    pages_blocks = {}
    for probe in probes:
        layout_type = layout_type_by_page[probe.page]
        if layout_type == "blank":
            pages_blocks[probe.page] = []
            continue
        if layout_type == "ruled_table":
            all_blocks = extract_text_blocks(probe, layouts[probe.page])
            table_bboxes = [t.bbox for t in table_blocks_by_page.get(probe.page, [])]
            pages_blocks[probe.page] = [
                b for b in all_blocks
                if not any(bbox["top"] - 2 <= b.top and b.bottom <= bbox["bottom"] + 2 for bbox in table_bboxes)
            ]
            continue
        pages_blocks[probe.page] = extract_text_blocks(probe, layouts[probe.page])

    page_order = sorted(p.page for p in probes)

    prelim_text_by_page = {page: prelim_page_text(pages_blocks.get(page, [])) for page in page_order}
    prelim_full_text = "\n\n".join(prelim_text_by_page.get(p, "") for p in page_order)

    layout_counts = Counter(lt for lt in layout_type_by_page.values() if lt != "blank")
    dominant_layout = layout_counts.most_common(1)[0][0] if layout_counts else "single_column"

    profile_list = profiles_mod.load_profiles(profile_dir or profiles_mod.DEFAULT_PROFILE_DIR)
    match = profiles_mod.select_profile(profile_list, prelim_full_text, len(probes), dominant_layout)

    sub_doc_by_block: dict[tuple[int, int], str | None] = {}
    segment_notes: list[str] = []
    if tree_engine == "relative":
        sub_doc_by_block, segment_notes = segments.assign_sub_documents_by_block(
            pages_blocks, page_order, match.profile,
            {p.page: p.width for p in probes},
        )
        sub_doc_by_page = segments.page_level_view(sub_doc_by_block, page_order) if sub_doc_by_block             else assign_sub_documents(page_order, prelim_text_by_page, match.profile)
    else:
        sub_doc_by_page = assign_sub_documents(page_order, prelim_text_by_page, match.profile)
    clause_sub_document = match.profile.get("expected_invariants", {}).get("clause_sequence_scope")
    vocab = vocabulary.for_profile(match.profile, str(profile_dir) if profile_dir else None)

    nodes, page_raw_text, tree_quality_flags = build_tree(
        pages_blocks, layout_type_by_page, page_order, sub_doc_by_page, clause_sub_document,
        vocab.get("running_header_patterns"),
        tree_engine,
        {p.page: p.width for p in probes},
        sub_doc_by_block,
    )

    for page, tblocks in table_blocks_by_page.items():
        flat = "\n".join(" | ".join(cell or "" for cell in row) for t in tblocks for row in t.rows)
        page_raw_text[page] = (page_raw_text.get(page, "") + "\n" + flat).strip()

    label_index = entities_mod.build_label_index(nodes)
    entities_mod.resolve_refs_out(nodes, label_index)
    entities_mod.tag_modality(nodes)

    full_text = "\n\n".join(page_raw_text.get(p, "") for p in page_order)
    document_status = guess_document_status(full_text)

    doc_entities = entities_mod.extract_document_entities(nodes, full_text)
    field_ctx = (
        field_context.FieldContext.from_pages(page_order, page_raw_text, sub_doc_by_page)
        if tree_engine == "relative" else None
    )
    core = core_fields.resolve_core(full_text, document_status, vocab, field_ctx)

    for n in nodes:
        if n.sub_document is None and not sub_doc_by_block:
            n.sub_document = sub_doc_by_page.get(n.pages[0]) if n.pages else None

    tables = build_table_entries(table_blocks_by_page, label_index)

    quality = run_validation(
        str(pdf_path), nodes, page_raw_text, layout_type_by_page, core, match.profile, tree_quality_flags,
        probes, route_decisions,
    )

    with pdfplumber.open(str(pdf_path)) as pdf:
        pdf_metadata = {k: str(v) for k, v in (pdf.metadata or {}).items()}

    pages_out = []
    for p in probes:
        raw_text = page_raw_text.get(p.page, "")
        pages_out.append(
            {
                "page": p.page,
                "page_label": extract_page_label(raw_text) if raw_text else None,
                "sub_document": sub_doc_by_page.get(p.page),
                "width": p.width,
                "height": p.height,
                "rotation": p.rotation,
                "extraction_method": route_by_page[p.page].method,
                "route_reason": route_by_page[p.page].reason,
                "layout_type": layout_type_by_page[p.page],
                "has_ruling_lines": p.ruling_line_count > 0,
                "raw_text": raw_text,
                "char_count": len(raw_text),
            }
        )

    structure_out = []
    for n in nodes:
        d = asdict(n)
        structure_out.append(d)

    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file": pdf_path.name,
            "sha256": sha256_of(pdf_path),
            "page_count": len(probes),
            "pdf_metadata": pdf_metadata,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "1.0.0",
            "tree_engine": tree_engine,
            "parsers": {"primary": "pdfplumber", "oracle": "poppler-pdftotext (if available)"},
        },
        "profile": {
            "profile_id": match.profile["profile_id"],
            "match_score": round(match.score, 3),
            "description": match.profile.get("description", ""),
        },
        "core": core,
        "pages": pages_out,
        "structure": structure_out,
        "tables": tables,
        "entities": doc_entities,
        "quality": quality,
    }
    return document


EXIT_PASSED, EXIT_ERROR, EXIT_VALIDATION_FAILED = 0, 1, 2


def _raw_sha(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("source", {}).get("sha256")
    except (OSError, ValueError):
        return None


def output_stem(out_dir: Path, pdf_path: Path, sha: str | None) -> str:
    existing = out_dir / f"{pdf_path.stem}_raw.json"
    if not sha or not existing.exists():
        return pdf_path.stem
    held = _raw_sha(existing)
    if held is None or held == sha:
        return pdf_path.stem
    return f"{pdf_path.stem}__{sha[:8]}"


def write_failure_record(out_dir: Path, pdf_path: Path, exc: BaseException, pipeline_version: str) -> Path:
    sha = sha256_of(pdf_path) if pdf_path.is_file() else None
    stem = output_stem(out_dir, pdf_path, sha)
    record = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file": pdf_path.name,
            "path": str(pdf_path),
            "sha256": sha,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": pipeline_version,
        },
        "pipeline_status": "failed",
        "error_class": type(exc).__name__,
        "error": str(exc),
    }
    out_path = out_dir / f"{stem}_status.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    (out_dir / f"{stem}_raw.json").unlink(missing_ok=True)
    return out_path


def write_document(out_dir: Path, pdf_path: Path, document: dict) -> Path:
    sha = document["source"]["sha256"]
    stem = output_stem(out_dir, pdf_path, sha)
    if stem != pdf_path.stem:
        document["source"]["display_name"] = f"{pdf_path.stem} ({sha[:8]}){pdf_path.suffix}"
        print(f"note: {pdf_path.stem}_raw.json holds a different PDF with the same name — "
              f"keeping it and writing this one as {stem}_raw.json")
    out_path = out_dir / f"{stem}_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    (out_dir / f"{stem}_status.json").unlink(missing_ok=True)
    return out_path


def print_quality(q: dict) -> None:
    print(f"validation: {q['pipeline_status']}  hard_fails={q['hard_fail_count']}  warns={q['warn_count']}")
    for c in q["checks"]:
        if c["status"] in ("fail", "warn"):
            print(f"  [{c['status'].upper()}] {c['check']}: {c['detail']}")


def run_batch(pdf_paths: list[Path], out_dir: Path, pipeline_version: str, process) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = []
    for pdf_path in pdf_paths:
        if len(pdf_paths) > 1:
            print(f"== {pdf_path}")
        try:
            if not pdf_path.is_file():
                raise FileNotFoundError(f"{pdf_path} does not exist")
            codes.append(process(pdf_path))
        except Exception as exc:
            record_path = write_failure_record(out_dir, pdf_path, exc, pipeline_version)
            print(f"error: {pdf_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"wrote {record_path}")
            codes.append(EXIT_ERROR)
    if len(pdf_paths) > 1:
        errors = codes.count(EXIT_ERROR)
        invalid = codes.count(EXIT_VALIDATION_FAILED)
        print(f"batch: {len(codes) - errors - invalid} passed, {invalid} failed validation, {errors} errored")
    if EXIT_ERROR in codes:
        return EXIT_ERROR
    return EXIT_VALIDATION_FAILED if EXIT_VALIDATION_FAILED in codes else EXIT_PASSED


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract born-digital Indonesian contract PDFs into raw_extraction.json")
    parser.add_argument("pdf_paths", type=Path, nargs="+", metavar="pdf_path", help="Path(s) to the input PDF(s)")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory (default: output/)")
    parser.add_argument("--profile-dir", type=Path, default=None, help="Override the profile directory")
    parser.add_argument("--tree-engine", choices=("legacy", "relative"), default=DEFAULT_TREE_ENGINE,
                        help=f"Tree depth engine (default: {DEFAULT_TREE_ENGINE})")
    args = parser.parse_args()

    def process(pdf_path: Path) -> int:
        document = run_pipeline(pdf_path, args.out, args.profile_dir, args.tree_engine)
        out_path = write_document(args.out, pdf_path, document)
        q = document["quality"]
        core_status = document["core"]["_status"]
        print(f"profile: {document['profile']['profile_id']} (score={document['profile']['match_score']})")
        print(f"pages: {document['source']['page_count']}  nodes: {len(document['structure'])}  tables: {len(document['tables'])}")
        print(f"core fields populated: {core_status['fields_populated']}/6  overall_confidence={core_status['overall_confidence']}")
        print_quality(q)
        print(f"wrote {out_path}")
        return EXIT_PASSED if q["pipeline_status"] == "passed" else EXIT_VALIDATION_FAILED

    return run_batch(args.pdf_paths, args.out, "1.0.0", process)


if __name__ == "__main__":
    raise SystemExit(main())
