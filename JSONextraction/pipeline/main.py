"""CLI orchestrator — runs Stages 1-9 end to end and writes raw_extraction.json.

Usage:
    python -m pipeline.main "Rancangan Kontrak.pdf" --out output/

v1 scope (see README.md): native text extraction only, no OCR engine and no
LLM fallback wired in. Pages that would need OCR are flagged, not silently
dropped. This produces `raw_extraction.json` — the source-of-truth layer.
Preprocessing/derivation (text_bm25, text_embed) and chunking are later
phases, not part of this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

from . import core_fields, entities as entities_mod, profiles as profiles_mod
from .blocks import extract_table_blocks, extract_text_blocks
from .layout import classify_layout
from .probe import probe_document
from .router import route_pages
from .schema import SCHEMA_VERSION
from .tree import build_tree
from .validate import run_validation

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
    """A cheap, tree-independent per-page text join used only to pick a
    profile and locate sub-document markers before the real tree exists.
    Order and content match what `build_tree` will later assemble into
    `page_raw_text` closely enough for marker regexes (heading lines) to
    match the same way — the two paths are never compared for equality."""
    return "\n".join(b.text for b in blocks)


def assign_sub_documents(page_order: list[int], page_raw_text: dict[int, str], profile: dict) -> dict[int, str | None]:
    """Finds each profile-declared marker's first occurrence (by page), then
    walks pages in order switching to whichever marker was first reached —
    in the ACTUAL page order of this document, not the profile's declaration
    order. A stray repeat match of an already-seen marker later on doesn't
    regress the current sub-document (only the first occurrence counts).

    This does NOT assume every document lays out its sections in the same
    order as the profile's `sub_document_markers` list: one real specimen
    binds "LAMPIRAN A/B" (annex_a/annex_b) right after the main agreement,
    before the general-terms/SSUK section — the reverse of the order
    perpres16_konstruksi_v1 declares (main_agreement, general_terms,
    special_terms, annex_a, annex_b). An earlier version of this function
    walked the marker list in strict declared order and could never revisit
    an earlier marker once a later one matched, so general_terms was
    permanently skipped for the rest of that document. Profiles with no
    markers (e.g. generic_contract_v1) leave every page's sub_document as
    None."""
    markers = profile.get("sub_document_markers", [])
    first_seen_page: dict[str, int] = {}
    for page in page_order:
        text = page_raw_text.get(page, "")
        for marker in markers:
            name = marker["name"]
            if name in first_seen_page:
                continue
            if re.search(marker["start"], text, re.IGNORECASE | re.MULTILINE):
                first_seen_page[name] = page

    events = sorted(first_seen_page.items(), key=lambda kv: kv[1])  # (name, first_page), by page
    result: dict[int, str | None] = {}
    current: str | None = None
    event_idx = 0
    for page in page_order:
        while event_idx < len(events) and events[event_idx][1] <= page:
            current = events[event_idx][0]
            event_idx += 1
        result[page] = current
    return result


def build_table_entries(table_blocks_by_page: dict[int, list], label_index: dict[str, str]) -> list[dict]:
    tables = []
    clause_ref_re = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){0,2}\b")
    for page, tblocks in sorted(table_blocks_by_page.items()):
        for t_idx, t in enumerate(tblocks):
            rows_out = []
            headers = t.rows[0] if t.rows else []
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


def run_pipeline(pdf_path: Path, output_dir: Path, profile_dir: Path | None = None) -> dict:
    probes = probe_document(str(pdf_path))
    route_decisions = route_pages(probes)
    route_by_page = {r.page: r for r in route_decisions}

    layouts = {p.page: classify_layout(p) for p in probes}
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
            # Table *cell* text goes to tables[], not the tree — but a page
            # can carry a title or caption outside the table's bbox (e.g. the
            # "SYARAT-SYARAT KHUSUS KONTRAK" heading above the SSKK table),
            # and dropping it entirely both loses that heading node and
            # starves sub-document marker detection of the text it needs.
            # Keep only the blocks that fall outside every table's bbox.
            all_blocks = extract_text_blocks(probe, layouts[probe.page])
            table_bboxes = [t.bbox for t in table_blocks_by_page.get(probe.page, [])]
            pages_blocks[probe.page] = [
                b for b in all_blocks
                if not any(bbox["top"] - 2 <= b.top and b.bottom <= bbox["bottom"] + 2 for bbox in table_bboxes)
            ]
            continue
        pages_blocks[probe.page] = extract_text_blocks(probe, layouts[probe.page])

    page_order = sorted(p.page for p in probes)

    # Profile selection and sub-document marker assignment need per-page text
    # but must run BEFORE build_tree, because tree.py needs to know which
    # pages are in the profile's clause-bearing sub-document (e.g.
    # "general_terms") to classify decimal_plain numbering as "clause" vs.
    # plain "list_item" (see tree.py's _classify). This prelim text is a
    # simple join, independent of tree construction — chicken-and-egg
    # avoided by not needing the tree to get it.
    prelim_text_by_page = {page: prelim_page_text(pages_blocks.get(page, [])) for page in page_order}
    prelim_full_text = "\n\n".join(prelim_text_by_page.get(p, "") for p in page_order)

    layout_counts = Counter(lt for lt in layout_type_by_page.values() if lt != "blank")
    dominant_layout = layout_counts.most_common(1)[0][0] if layout_counts else "single_column"

    profile_list = profiles_mod.load_profiles(profile_dir or profiles_mod.DEFAULT_PROFILE_DIR)
    match = profiles_mod.select_profile(profile_list, prelim_full_text, len(probes), dominant_layout)

    sub_doc_by_page = assign_sub_documents(page_order, prelim_text_by_page, match.profile)
    clause_sub_document = match.profile.get("expected_invariants", {}).get("clause_sequence_scope")

    nodes, page_raw_text, tree_quality_flags = build_tree(
        pages_blocks, layout_type_by_page, page_order, sub_doc_by_page, clause_sub_document
    )

    # Fold ruled-table cell text into page_raw_text so entity/core regexes and
    # char-conservation bookkeeping can see it (tables live in tables[], not
    # in the node tree, but their text is still part of the document).
    for page, tblocks in table_blocks_by_page.items():
        flat = "\n".join(" | ".join(cell or "" for cell in row) for t in tblocks for row in t.rows)
        page_raw_text[page] = (page_raw_text.get(page, "") + "\n" + flat).strip()

    label_index = entities_mod.build_label_index(nodes)
    entities_mod.resolve_refs_out(nodes, label_index)
    entities_mod.tag_modality(nodes)

    full_text = "\n\n".join(page_raw_text.get(p, "") for p in page_order)
    document_status = guess_document_status(full_text)

    doc_entities = entities_mod.extract_document_entities(nodes, full_text)
    core = core_fields.resolve_core(full_text, document_status)

    for n in nodes:
        n.sub_document = sub_doc_by_page.get(n.pages[0]) if n.pages else None

    tables = build_table_entries(table_blocks_by_page, label_index)

    quality = run_validation(
        str(pdf_path), nodes, page_raw_text, layout_type_by_page, core, match.profile, tree_quality_flags,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a born-digital Indonesian contract PDF into raw_extraction.json")
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory (default: output/)")
    parser.add_argument("--profile-dir", type=Path, default=None, help="Override the profile directory")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"error: {args.pdf_path} does not exist", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    document = run_pipeline(args.pdf_path, args.out, args.profile_dir)

    out_path = args.out / "raw_extraction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    q = document["quality"]
    core_status = document["core"]["_status"]
    print(f"profile: {document['profile']['profile_id']} (score={document['profile']['match_score']})")
    print(f"pages: {document['source']['page_count']}  nodes: {len(document['structure'])}  tables: {len(document['tables'])}")
    print(f"core fields populated: {core_status['fields_populated']}/6  overall_confidence={core_status['overall_confidence']}")
    print(f"validation: {q['pipeline_status']}  hard_fails={q['hard_fail_count']}  warns={q['warn_count']}")
    for c in q["checks"]:
        if c["status"] in ("fail", "warn"):
            print(f"  [{c['status'].upper()}] {c['check']}: {c['detail']}")
    print(f"wrote {out_path}")
    return 0 if q["pipeline_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
