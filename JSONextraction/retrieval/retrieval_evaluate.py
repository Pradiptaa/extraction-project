from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_settings
from .embed import Embedder
from .retrievers import DenseRetriever, build_retriever
from .schema import EMBEDDING_SCHEMA_VERSION
from .store import open_collection 

logger = logging.getLogger("retrieval.evaluate")

QUERY_SCHEMA_VERSION = "1.0.0"
BASELINE_SCHEMA_VERSION = "1.0.0"

DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "ground_truth" / "retrieval_baseline.json"
EQUIVALENCE_MIN_CHARS = 60


def clause_key(metadata: dict) -> tuple[str, str, str]:
    return (
        metadata.get("sub_document") or "",
        metadata.get("hierarchy_path") or "",
        metadata.get("label") or "",
    )


def match_kind(metadata: dict, expect: dict) -> str | None:
    if (metadata.get("sub_document") or "") != (expect.get("sub_document") or ""):
        return None

    path = metadata.get("hierarchy_path") or ""
    target = expect.get("hierarchy_path") or ""

    if path == target:
        return "exact"

    collapsed = _collapse_subdivisions(path)
    if target and collapsed == target:
        return "refined"
    if target and (path.startswith(target + "/") or collapsed.startswith(target + "/")):
        return "descendant"
    return None


def _collapse_subdivisions(path: str) -> str:
    kept: list[str] = []
    for segment in (s for s in path.split("/") if s):
        previous = kept[-1] if kept else ""
        if previous and previous[0].isalpha() and segment.startswith(previous + "."):
            continue
        kept.append(segment)
    return "/".join(kept)


@dataclass
class QueryResult:
    query_id: str
    query: str
    passed: bool
    rank: int | None
    detail: str
    match_kind: str | None = None
    class_size: int = 0
    documents_hit: list[str] = field(default_factory=list)
    top: list[tuple[str, float, str]] = field(default_factory=list)


def load_queries(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))

    version = spec.get("embedding_schema_version")
    if version != EMBEDDING_SCHEMA_VERSION:
        raise SystemExit(
            f"{path.name} targets embedding schema {version}, this build is "
            f"{EMBEDDING_SCHEMA_VERSION}. Re-check the query set before scoring against it."
        )

    queries = spec.get("queries") or []
    if not queries:
        raise SystemExit(f"{path.name} defines no queries")

    seen = set()
    for q in queries:
        for required in ("id", "query", "expect"):
            if not q.get(required):
                raise SystemExit(f"query {q.get('id', '?')!r} is missing {required!r}")
        if not (q["expect"].get("hierarchy_path") or q["expect"].get("text_contains")):
            raise SystemExit(f"query {q['id']!r} expect needs hierarchy_path or text_contains")
        if q.get("expect_ref") and not (q["expect_ref"].get("sub_document") and q["expect_ref"].get("path_suffix")):
            raise SystemExit(f"query {q['id']!r} expect_ref needs sub_document and path_suffix")
        if q["id"] in seen:
            raise SystemExit(f"duplicate query id {q['id']!r}")
        seen.add(q["id"])

    return spec


def build_class_index(collection) -> dict[tuple[str, str, str], list[dict]]:
    got = collection.get(include=["metadatas", "documents"])
    documents = got.get("documents") or [""] * len(got["ids"])
    index: dict[tuple[str, str, str], list[dict]] = {}
    for row_id, metadata, text in zip(got["ids"], got["metadatas"], documents):
        entry = dict(metadata or {})
        entry["_id"] = row_id
        entry["_text"] = text or ""
        index.setdefault(clause_key(entry), []).append(entry)
    return index


def accepted_rows(class_index: dict, expect: dict) -> dict[str, str]:
    if expect.get("hierarchy_path"):
        accepted = _clause_rows(class_index, expect)
    elif expect.get("text_contains"):
        accepted = {
            row["_id"]: "exact"
            for rows in class_index.values() for row in rows
            if (row.get("sub_document") or "") == (expect.get("sub_document") or "")
        }
    else:
        return {}

    if expect.get("node_type") or expect.get("text_contains"):
        rows_by_id = {row["_id"]: row for rows in class_index.values() for row in rows}
        accepted = {
            row_id: kind for row_id, kind in accepted.items()
            if (not expect.get("node_type") or rows_by_id[row_id].get("node_type") == expect["node_type"])
            and (not expect.get("text_contains") or expect["text_contains"] in rows_by_id[row_id]["_text"])
        }
    return accepted


def ref_check(metadata: dict, expect_ref: dict) -> bool:
    suffix = expect_ref["path_suffix"]
    for target in (metadata.get("ref_targets") or "").split(";"):
        sub_document, _, path = target.partition(":")
        if sub_document != expect_ref["sub_document"]:
            continue
        if path == suffix or path.endswith("/" + suffix):
            return True
    return False


def _clause_rows(class_index: dict, expect: dict) -> dict[str, str]:
    accepted: dict[str, str] = {}
    for rows in class_index.values():
        for row in rows:
            kind = match_kind(row, expect)
            if kind:
                accepted[row["_id"]] = kind

    texts = {row["_text"] for rows in class_index.values() for row in rows
             if row["_id"] in accepted and row["_text"]}
    if texts:
        for rows in class_index.values():
            for row in rows:
                if row["_id"] in accepted or row["_text"] not in texts:
                    continue
                if (len(row["_text"]) >= EQUIVALENCE_MIN_CHARS
                        or _same_clause_ignoring_section_letter(row, expect)):
                    accepted[row["_id"]] = "equivalent"
    return accepted


def _strip_section_letter(path: str) -> str:
    segments = path.split("/")
    if len(segments) > 1 and len(segments[0]) == 1 and segments[0].isalpha() and segments[0].isupper():
        segments = segments[1:]
    return "/".join(segments)


def _same_clause_ignoring_section_letter(row: dict, expect: dict) -> bool:
    row_view = dict(row, hierarchy_path=_strip_section_letter(row.get("hierarchy_path") or ""))
    expect_view = dict(expect, hierarchy_path=_strip_section_letter(expect.get("hierarchy_path") or ""))
    return match_kind(row_view, expect_view) is not None


def evaluate_query(
    spec: dict,
    retriever,
    class_index: dict[tuple[str, str, str], list[dict]],
    k: int,
    max_rank: int | None,
) -> QueryResult:
    expect = spec["expect"]
    key = tuple(
        expect[field] for field in ("sub_document", "hierarchy_path", "label", "node_type", "text_contains")
        if expect.get(field)
    )

    accepted = accepted_rows(class_index, expect)
    exact_rows = {row_id for row_id, kind in accepted.items() if kind in ("exact", "refined")}

    if not accepted:
        return QueryResult(
            query_id=spec["id"],
            query=spec["query"],
            passed=False,
            rank=None,
            detail=f"expected clause {key} is not in the collection at all",
        )

    hits = retriever.search(spec["query"], k)
    ids = [hit.id for hit in hits]
    distances = [hit.score for hit in hits]
    metadatas = [hit.metadata for hit in hits]

    top = [(hit.id, hit.score, " ".join((hit.text or "").split())[:70]) for hit in hits]

    rank = None
    kind = None
    documents_hit: list[str] = []
    for position, (row_id, metadata) in enumerate(zip(ids, metadatas), start=1):
        if row_id in accepted:
            if rank is None:
                rank = position
                kind = accepted[row_id]
            documents_hit.append((metadata or {}).get("document_key", "")[:8])

    exact_hits = sum(1 for row_id in ids if row_id in exact_rows)

    expect_ref = spec.get("expect_ref")
    ref_ok = expect_ref is None or any(
        ref_check(metadata or {}, expect_ref)
        for row_id, metadata in zip(ids, metadatas) if row_id in accepted
    )

    if rank is None:
        best = f"best was [{distances[0]:.3f}] {clause_key(metadatas[0])}" if ids else "no results"
        detail = (
            f"neither the clause nor any sub-clause in top-{k} "
            f"({len(accepted)} acceptable rows, {len(exact_rows)} exact); {best}"
        )
        passed = False
    elif max_rank is not None and rank > max_rank:
        detail = f"found at rank {rank} ({kind}), worse than --max-rank {max_rank}"
        passed = False
    elif not ref_ok:
        refs = sorted({(metadata or {}).get("ref_targets") or "(none)"
                       for row_id, metadata in zip(ids, metadatas) if row_id in accepted})
        detail = (
            f"found at rank {rank} ({kind}), but no accepted hit references "
            f"{expect_ref['sub_document']}:…/{expect_ref['path_suffix']} — refs were {refs}"
        )
        passed = False
    else:
        detail = (
            f"rank {rank} of {k} ({kind}), {len(documents_hit)} accepted rows in top-k"
            f", {exact_hits} of them the clause itself"
        )
        passed = True

    return QueryResult(
        query_id=spec["id"],
        query=spec["query"],
        passed=passed,
        rank=rank,
        detail=detail,
        match_kind=kind if passed else None,
        class_size=len(accepted),
        documents_hit=documents_hit,
        top=top,
    )


def run(
    spec: dict,
    collection,
    embedder,
    k: int,
    max_rank: int | None = None,
    verbose: bool = False,
    retriever=None,
    expected_pass: set[str] | None = None,
) -> tuple[bool, list[QueryResult]]:
    if retriever is None:
        retriever = DenseRetriever(collection, embedder)

    class_index = build_class_index(collection)
    results = []

    for query_spec in spec["queries"]:
        query_k = int(query_spec.get("k") or k)
        result = evaluate_query(query_spec, retriever, class_index, query_k, max_rank)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.query_id}: {result.detail}")
        print(f"         query: {result.query}")
        if verbose or not result.passed:
            for position, (row_id, dist, text) in enumerate(result.top, start=1):
                marker = "*" if result.rank == position else " "
                print(f"       {marker} {position}. [{dist:.3f}] {row_id} {text}")

    passed = sum(1 for r in results if r.passed)
    print()
    print(f"{passed}/{len(results)} retrieval checks passed")

    kinds = Counter(r.match_kind for r in results if r.passed and r.match_kind)
    if kinds:
        print("  by match kind: " + ", ".join(f"{kind}={n}" for kind, n in sorted(kinds.items())))
    print()

    if expected_pass is None:
        ok = passed == len(results)
        print("RESULT: PASS" if ok else "RESULT: FAIL")
        return ok, results

    regressions, improvements = compare_to_baseline(results, expected_pass)
    print(f"baseline   : {len(expected_pass)} expected to pass")
    for query_id in regressions:
        print(f"  [REGRESSION] {query_id} passed in the baseline and fails now")
    for query_id in improvements:
        print(f"  [NEW PASS]   {query_id} fails in the baseline and passes now "
              "— say which change caused it before recording it with --update-baseline")
    ok = not regressions
    print("RESULT: PASS (no regressions against the baseline)" if ok
          else f"RESULT: FAIL ({len(regressions)} regression(s) against the baseline)")
    return ok, results


def compare_to_baseline(results: list[QueryResult], expected_pass: set[str]) -> tuple[list[str], list[str]]:
    current = {r.query_id for r in results}
    passing = {r.query_id for r in results if r.passed}
    regressions = sorted((expected_pass & current) - passing)
    improvements = sorted(passing - expected_pass)
    return regressions, improvements


def baseline_key(retriever: str, tokenizer: str, k: int) -> str:
    if retriever in ("bm25", "hybrid", "hybrid-brute"):
        return f"{retriever}/{tokenizer}/k={k}"
    return f"{retriever}/k={k}"


def load_baseline(path: Path, key: str, collection: str) -> set[str] | None:
    if not path.exists():
        logger.warning("no baseline file at %s — using the strict verdict", path)
        return None
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise SystemExit(
            f"{path.name} is baseline schema {spec.get('schema_version')}, expected {BASELINE_SCHEMA_VERSION}"
        )
    entry = (spec.get("baselines") or {}).get(key)
    if entry is None:
        logger.warning("%s has no baseline for %s — using the strict verdict", path.name, key)
        return None
    if entry.get("collection") != collection:
        logger.warning(
            "baseline for %s was recorded on collection %r, not %r — it does not apply; "
            "using the strict verdict. Record a new one with --update-baseline once the "
            "score on this collection is understood.",
            key, entry.get("collection"), collection,
        )
        return None
    return set(entry.get("passed") or [])


def write_baseline(path: Path, key: str, collection: str, results: list[QueryResult], recorded: str) -> None:
    spec = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema_version": BASELINE_SCHEMA_VERSION, "baselines": {}}
    )
    spec.setdefault("baselines", {})[key] = {
        "collection": collection,
        "recorded": recorded,
        "total": len(results),
        "passed": sorted(r.query_id for r in results if r.passed),
    }
    spec["baselines"] = dict(sorted(spec["baselines"].items()))
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Score retrieval against the ground-truth query set")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ground_truth" / "retrieval_queries.json",
    )
    parser.add_argument("-k", type=int, default=None, help="Top-k to consider (default: the set's default_k)")
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Also fail a query whose expected clause ranks worse than this, to catch degradation within top-k",
    )
    parser.add_argument("--verbose", action="store_true", help="Show the top-k for passing queries too")
    parser.add_argument(
        "--retriever",
        choices=("dense", "brute", "bm25", "hybrid", "hybrid-brute"),
        default="hybrid",
        help="Which strategy to score (default: hybrid, the same default as `retrieval.ask`, so the "
             "gate measures what users get). `brute` and `hybrid-brute` are exact-search reference "
             "ceilings, not production strategies",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=50,
        help="Candidates each side contributes before fusion (hybrid only). Not a quality knob — see HybridRetriever",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("plain", "nostop", "stem"),
        default="plain",
        help="Lexical tokenisation for bm25/hybrid",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Recorded expected-pass sets; the run fails only on a regression against it",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Ignore the baseline and require every query to pass",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record this run's passes as the baseline for this configuration. Only after "
             "the change in score has been explained",
    )
    args = parser.parse_args()

    if not args.queries.exists():
        logger.error("no such query set: %s", args.queries)
        return 1

    spec = load_queries(args.queries)
    k = args.k or int(spec.get("default_k", 5))

    settings = load_settings()
    collection = open_collection(settings)
    embedder = Embedder(settings.model, settings.host, num_gpu=settings.query_num_gpu)
    retriever = build_retriever(args.retriever, collection, embedder, args.pool, args.tokenizer)

    print(f"collection : {settings.collection} ({collection.count()} rows)")
    print(f"model      : {settings.model}")
    print(f"retriever  : {args.retriever}" + (
        f" (pool={args.pool}, tokenizer={args.tokenizer})" if args.retriever == "hybrid"
        else f" (tokenizer={args.tokenizer})" if args.retriever == "bm25" else ""
    ))
    print(f"queries    : {args.queries.name} ({len(spec['queries'])} queries, k={k})")

    key = baseline_key(args.retriever, args.tokenizer, k)
    expected_pass = None
    if not (args.strict or args.update_baseline or args.max_rank is not None):
        expected_pass = load_baseline(args.baseline, key, settings.collection)
    elif args.max_rank is not None and not args.strict:
        logger.warning("--max-rank is not covered by the baseline — using the strict verdict")
    print(f"verdict    : " + (f"regressions against {args.baseline.name} [{key}]"
                              if expected_pass is not None else "strict (every query must pass)"))
    print()

    ok, results = run(
        spec, collection, embedder, k, args.max_rank, args.verbose,
        retriever=retriever, expected_pass=expected_pass,
    )

    if args.update_baseline:
        from datetime import date

        write_baseline(args.baseline, key, settings.collection, results, date.today().isoformat())
        passed = sum(1 for r in results if r.passed)
        print(f"\nbaseline [{key}] recorded: {passed}/{len(results)} -> {args.baseline}")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
