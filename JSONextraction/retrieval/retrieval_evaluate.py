"""Regression gate for the retrieval layer.

Counterpart to `pipeline/evaluate.py`: that scores extraction against
per-specimen ground truth, this scores retrieval against a fixed query set.
Same contract — per-check PASS/FAIL lines, a summary, and exit 0 only if every
check passed. From here on, retrieval changes are judged by this, not by
eyeballing a few queries.

    python -m retrieval.retrieval_evaluate --queries ground_truth/retrieval_queries.json
    python -m retrieval.retrieval_evaluate -k 10 --verbose

A query names a target CLAUSE, and passes when the top-k contains any row that
answers it. Two deliberate leniencies define "answers it", and both are about
the question being asked of a corpus, not about making the gate easy:

1. Any document's copy counts. Every specimen is the same Perpres 16/2018
   standard form, so one clause exists in several of them at nearly identical
   distance. Which contract the answer came from is not what a corpus-wide
   semantic query is asking.
2. Any sub-clause beneath the target counts. A clause node holds a heading and
   lead text; the provision a question is really about usually sits in a
   numbered child. See `match_kind` — the relation is one-directional, so a
   descendant passes and an ancestor does not.

Both are bounded. A different clause fails, an ancestor fails, and the report
says whether the hit was the clause itself or a descendant, so a drift from
exact to descendant hits stays visible rather than being absorbed.

Rank is reported, not just hit/miss. A target sliding from rank 1 to rank 4 is
a real degradation that a boolean would swallow, and `--max-rank` can turn it
into a failure once a baseline is established.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import Settings, load_settings
from .embed import Embedder
from .schema import EMBEDDING_SCHEMA_VERSION

logger = logging.getLogger("retrieval.evaluate")

QUERY_SCHEMA_VERSION = "1.0.0"


def clause_key(metadata: dict) -> tuple[str, str, str]:
    """The document-agnostic identity of a clause.

    These are exactly the `embedding_id` components that do NOT vary between
    two contracts holding the same standard clause. `document_key`, `page`,
    `text` and `occurrence` are excluded precisely because they do vary — see
    `retrieval/schema.py`.

    Note this key is intentionally ambiguous WITHIN a document (several lists
    restart at "1." under one sub-document, so they share a path). That is
    harmless here: it can only widen an equivalence class, and a wider class is
    a more lenient check, never a false pass on an unrelated clause.
    """
    return (
        metadata.get("sub_document") or "",
        metadata.get("hierarchy_path") or "",
        metadata.get("label") or "",
    )


def match_kind(metadata: dict, expect: dict) -> str | None:
    """How a returned row relates to the expected clause: exact, descendant, or
    no relation.

    Descendants count as hits. A clause node carries only its heading and lead
    text, while the provision a question is actually about usually sits in a
    numbered sub-clause beneath it — clause 55 is "Asuransi", and 55.2 is the
    sentence obliging the provider to insure third parties. Returning 55.2 for
    "kewajiban penyedia mengasuransikan" is a better answer than the heading,
    so scoring it as a miss would train the retrieval layer away from its most
    useful behaviour.

    Ancestry is decided by path prefix within one `sub_document`, not by a
    parent-id chain, because Chroma metadata is flat. That is sound: a path is
    the chain of labels from the section root down, so a descendant's path
    necessarily begins with its ancestor's. The segment boundary matters —
    "C/6" must not swallow "C/61", which is a different clause entirely.

    The match is deliberately one-directional. A descendant passes; an ANCESTOR
    does not. Returning the whole section when asked about one clause is a real
    loss of precision, not a near-miss.
    """
    if (metadata.get("sub_document") or "") != (expect.get("sub_document") or ""):
        return None

    path = metadata.get("hierarchy_path") or ""
    target = expect.get("hierarchy_path") or ""

    if path == target and (metadata.get("label") or "") == (expect.get("label") or ""):
        return "exact"
    if path == target:
        # Same node position, different label: treat as exact rather than
        # inventing a third category — the label is a formatting detail.
        return "exact"
    if target and path.startswith(target + "/"):
        return "descendant"
    return None


@dataclass
class QueryResult:
    query_id: str
    query: str
    passed: bool
    rank: int | None
    detail: str
    class_size: int = 0
    documents_hit: list[str] = field(default_factory=list)
    top: list[tuple[str, float, str]] = field(default_factory=list)


def load_queries(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))

    version = spec.get("embedding_schema_version")
    if version != EMBEDDING_SCHEMA_VERSION:
        # The query set names clauses, not raw ids, so it mostly survives a
        # schema bump — but the collection it is scored against is chosen by
        # that version, so a mismatch means the two are about different data.
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
        if q["id"] in seen:
            raise SystemExit(f"duplicate query id {q['id']!r}")
        seen.add(q["id"])

    return spec


def build_class_index(collection) -> dict[tuple[str, str, str], list[dict]]:
    """Map every clause key in the collection to the rows that carry it.

    Read once up front rather than per query: the corpus is a few thousand rows,
    and doing it here means a query whose expected clause is absent from the
    collection is reported as a bad expectation rather than as a retrieval miss.
    """
    got = collection.get(include=["metadatas"])
    index: dict[tuple[str, str, str], list[dict]] = {}
    for row_id, metadata in zip(got["ids"], got["metadatas"]):
        entry = dict(metadata or {})
        entry["_id"] = row_id
        index.setdefault(clause_key(entry), []).append(entry)
    return index


def evaluate_query(
    spec: dict,
    collection,
    embedder,
    class_index: dict[tuple[str, str, str], list[dict]],
    k: int,
    max_rank: int | None,
) -> QueryResult:
    expect = spec["expect"]
    key = (
        expect.get("sub_document") or "",
        expect.get("hierarchy_path") or "",
        expect.get("label") or "",
    )

    # The acceptable set is the clause itself in every document that holds it,
    # plus everything beneath it. `exact_rows` is tracked separately so the
    # report can say which kind of hit was found.
    exact_rows = {row["_id"] for row in class_index.get(key, [])}
    accepted: dict[str, str] = {}
    for rows in class_index.values():
        for row in rows:
            kind = match_kind(row, expect)
            if kind:
                accepted[row["_id"]] = kind

    if not accepted:
        # Not a retrieval failure. The query set points at a clause the
        # collection does not contain, which is a stale expectation or an
        # incomplete load, and saying "miss" would misattribute the cause.
        return QueryResult(
            query_id=spec["id"],
            query=spec["query"],
            passed=False,
            rank=None,
            detail=f"expected clause {key} is not in the collection at all",
        )

    vector = embedder.embed([spec["query"]])[0]
    hits = collection.query(
        query_embeddings=[vector],
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )

    ids = hits["ids"][0]
    distances = hits["distances"][0]
    metadatas = hits["metadatas"][0]
    documents = hits.get("documents", [[]])[0] or [""] * len(ids)

    top = [
        (row_id, dist, " ".join((text or "").split())[:70])
        for row_id, dist, text in zip(ids, distances, documents)
    ]

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
) -> tuple[bool, list[QueryResult]]:
    class_index = build_class_index(collection)
    results = []

    for query_spec in spec["queries"]:
        query_k = int(query_spec.get("k") or k)
        result = evaluate_query(query_spec, collection, embedder, class_index, query_k, max_rank)
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
    print()
    print("RESULT: PASS" if passed == len(results) else "RESULT: FAIL")
    return passed == len(results), results


def open_collection(settings: Settings):
    if not settings.db_path.exists():
        raise SystemExit(
            f"no Chroma store at {settings.db_path} — run `python -m retrieval.load` first"
        )
    client = chromadb.PersistentClient(path=str(settings.db_path))
    try:
        collection = client.get_collection(settings.collection)
    except Exception:
        available = [c.name for c in client.list_collections()]
        raise SystemExit(
            f"collection {settings.collection!r} does not exist. Present: {available or 'none'}. "
            "Run `python -m retrieval.load` for this model and schema version."
        )
    if collection.count() == 0:
        # Distinguished from a retrieval failure on purpose: an empty collection
        # would fail every query and read as a catastrophic regression.
        raise SystemExit(
            f"collection {settings.collection!r} is empty — nothing has been loaded into it"
        )
    return collection


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
    args = parser.parse_args()

    if not args.queries.exists():
        logger.error("no such query set: %s", args.queries)
        return 1

    spec = load_queries(args.queries)
    k = args.k or int(spec.get("default_k", 5))

    settings = load_settings()
    collection = open_collection(settings)
    embedder = Embedder(settings.api_key, settings.model, settings.request_delay)

    print(f"collection : {settings.collection} ({collection.count()} rows)")
    print(f"model      : {settings.model}")
    print(f"queries    : {args.queries.name} ({len(spec['queries'])} queries, k={k})")
    print()

    ok, _ = run(spec, collection, embedder, k, args.max_rank, args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
