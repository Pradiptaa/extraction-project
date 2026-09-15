"""Regression gate for the retrieval layer.

Counterpart to `pipeline/evaluate.py`: that scores extraction against
per-specimen ground truth, this scores retrieval against a fixed query set.
Same contract — per-check PASS/FAIL lines, a summary, and exit 0 when the
system is at least as good as its recorded baseline
(`ground_truth/retrieval_baseline.json`, one entry per retriever/tokenizer/k).
Exit 1 means a query the baseline passed now fails. `--strict` restores "every
query must pass". From here on, retrieval changes are judged by this, not by
eyeballing a few queries.

    python -m retrieval.retrieval_evaluate                      # hybrid, vs baseline
    python -m retrieval.retrieval_evaluate --retriever dense --verbose
    python -m retrieval.retrieval_evaluate --strict
    python -m retrieval.retrieval_evaluate --update-baseline    # only once a score change is explained

A query names a target CLAUSE, and passes when the top-k contains any row that
answers it. Three deliberate leniencies define "answers it", and each is about
the question being asked of a corpus, not about making the gate easy:

1. Any document's copy counts. Every specimen is the same Perpres 16/2018
   standard form, so one clause exists in several of them at nearly identical
   distance. Which contract the answer came from is not what a corpus-wide
   semantic query is asking.
2. Any sub-clause beneath the target counts. A clause node holds a heading and
   lead text; the provision a question is really about usually sits in a
   numbered child. See `match_kind` — the relation is one-directional, so a
   descendant passes and an ancestor does not.
3. A row whose text is byte-identical to an accepted row counts. Without this
   the score swings by 4 queries on arbitrary tie-ordering alone, because the
   same sentence carries two different clause keys across specimens. See
   `accepted_rows` for the measurement, and for why this is not the kind
   of workaround that would hide the underlying bug.

All three are bounded. A different clause fails, an ancestor fails, different
text fails, and every hit is reported as `exact`, `descendant` or `equivalent`,
so a drift between them stays visible rather than being absorbed.

Rank is reported, not just hit/miss. A target sliding from rank 1 to rank 4 is
a real degradation that a boolean would swallow, and `--max-rank` can turn it
into a failure once a baseline is established.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_settings, needs_api_key
from .embed import Embedder
from .retrievers import DenseRetriever, build_retriever
from .schema import EMBEDDING_SCHEMA_VERSION
from .store import open_collection  # noqa: F401 — re-exported; tests and older callers import it from here

logger = logging.getLogger("retrieval.evaluate")

QUERY_SCHEMA_VERSION = "1.0.0"
BASELINE_SCHEMA_VERSION = "1.0.0"

DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "ground_truth" / "retrieval_baseline.json"

# Below this length, byte-identical text is not evidence of the same provision.
# Measured 2026-09-15 over the 206 texts that occur under more than one clause
# number: the short ones are generic list items and headings — "Bank Umum;",
# "Pengadilan.", "pemutusan Kontrak;", "perubahan pekerjaan;" — while nothing
# of 60+ chars was a coincidental match. Short rows still qualify when the path,
# with its section letter stripped, puts them under the target clause (see
# `_same_clause_ignoring_section_letter`), which is the case the equivalence
# rule exists for. Every retriever's gate score was unchanged by this bound
# (dense 11, bm25 13, hybrid 13); it removed 40 unrelated fragments from the
# accepted sets, which were false passes waiting to happen.
EQUIVALENCE_MIN_CHARS = 60


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

    if path == target:
        # The label is not compared: at the same node position a differing
        # label is a formatting detail, not a different clause.
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
    match_kind: str | None = None
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
        if not (q["expect"].get("hierarchy_path") or q["expect"].get("text_contains")):
            # Without either, `accepted_rows` has nothing to match on, and an
            # empty target would read as "clause not in the collection".
            raise SystemExit(f"query {q['id']!r} expect needs hierarchy_path or text_contains")
        if q.get("expect_ref") and not (q["expect_ref"].get("sub_document") and q["expect_ref"].get("path_suffix")):
            raise SystemExit(f"query {q['id']!r} expect_ref needs sub_document and path_suffix")
        if q["id"] in seen:
            raise SystemExit(f"duplicate query id {q['id']!r}")
        seen.add(q["id"])

    return spec


def build_class_index(collection) -> dict[tuple[str, str, str], list[dict]]:
    """Map every clause key in the collection to the rows that carry it.

    Read once up front rather than per query: the corpus is a few thousand rows,
    and doing it here means a query whose expected clause is absent from the
    collection is reported as a bad expectation rather than as a retrieval miss.

    Each row's text is carried along as `_text` because the accepted set is
    widened by text equality — see `accepted_rows`.
    """
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
    """Every row id that answers `expect`, mapped to how it qualifies.

    Two kinds of target:

    - A CLAUSE (`hierarchy_path` given): the clause rules in
      `_clause_rows` — exact, descendant, or text-equivalent.
    - A ROW BY CONTENT (no `hierarchy_path`, `text_contains` given): every row
      in `sub_document` whose text contains the string, all `exact`. This is
      how SSKK table rows are targeted: their path is a positional table id
      (`t_062_0/0`) that differs per specimen and says nothing about content.

    Two optional narrowing filters apply to both:

    - `node_type`: e.g. only `table_row`, so the SSUK clause that shares a
      heading with an SSKK row cannot answer a question about the SSKK value.
    - `text_contains` on a clause target: the accepted row must also hold this
      exact substring. That turns "the right clause came back" into "the right
      clause came back with this identifier or typo intact", which is the
      identifier-survival check run through retrieval rather than by hand.
      Byte-exact, never normalised — normalising is precisely the corruption
      it exists to catch.
    """
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
    """Whether a table row's cross-references include the expected target.

    `ref_targets` is "sub_document:path;..." (see `load._metadata`). The path is
    matched by SUFFIX on a segment boundary, so `4/4.1` accepts both `A/4/4.1`
    and the letterless `4/4.1` — the known clause-path inconsistency is about
    where the section letter appears, not about which clause a reference means.
    """
    suffix = expect_ref["path_suffix"]
    for target in (metadata.get("ref_targets") or "").split(";"):
        sub_document, _, path = target.partition(":")
        if sub_document != expect_ref["sub_document"]:
            continue
        if path == suffix or path.endswith("/" + suffix):
            return True
    return False


def _clause_rows(class_index: dict, expect: dict) -> dict[str, str]:
    """Every row id that answers a clause `expect`, mapped to how it qualifies:
    `exact`, `descendant`, or `equivalent`.

    The first two come from `match_kind`. The third exists because the corpus
    contains the same sentence under two different clause keys, and without it
    this gate does not measure retrieval.

    Measured: scoring one fixed retrieval config under 8 arbitrary tie-orderings
    of the corpus produced scores from 9/16 to 13/16 — a spread wider than any
    plausible difference between two retrievers. 60% of rows are duplicate text,
    and for nearly every query there are about as many byte-identical rows
    OUTSIDE the accepted set as inside it (q02: 58 in, 62 out). So which of two
    identical sentences the retriever happened to return decided pass/fail.

    The upstream cause is a clause-path inconsistency across specimens: the
    section letter is part of `hierarchy_path` in two specimens (`C/55`) and
    absent in the other two (`55`), so one standard clause carries two keys.

    §7 says not to work around that in the harness, and that rule still holds
    for the *clause key* — `expect_documents` still records 2 and stays honest.
    This is a different thing: it does not loosen what counts as the target
    clause, it says that a row whose text is byte-identical to an accepted row
    is the same answer. A reader handed that row cannot tell the difference,
    because there is no difference. Nothing about the path is relaxed, and a row
    with different text still fails.

    Widening is bounded by exact text equality — not by prefix, similarity, or
    normalisation — and `equivalent` is reported separately from `exact`.

    Note what that separate count can and cannot show. The pass/fail verdict is
    tie-stable; the KIND of a pass is not, because it is read off whichever of
    several byte-identical rows ranked first. Measured 2026-09-15: `dense` and
    `brute` passed the same 13 queries while reporting exact=10/descendant=3 and
    equivalent=6/exact=6/descendant=1. So the count is not a measure of the
    clause-path bug; compare pass sets.

    A second bound applies to short text. Identical short strings are common
    across unrelated clauses ("Pengadilan.", "Bank Umum;"), so a row under
    `EQUIVALENCE_MIN_CHARS` only qualifies if its path, with a leading section
    letter stripped, still places it under the target clause. Long rows need no
    structural agreement: at least one specimen has a third, broken path shape
    (`B/B.5/1.120` for a clause 41 sentence), and a 100-char identical sentence
    is the same provision whatever path it was filed under.
    """
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
    """`C/55/55.2` and `55/55.2` are the same position — the known clause-path
    inconsistency. Used only to corroborate a short text match, never as a
    match rule on its own, so the clause key itself stays strict."""
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

    # The acceptable set — see `accepted_rows`. `exact_rows` is tracked
    # separately so the report can say which kind of hit was found.
    accepted = accepted_rows(class_index, expect)
    exact_rows = {row_id for row_id, kind in accepted.items() if kind == "exact"}

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

    # The cross-reference check: an SSKK row that came back must still point
    # at its SSUK clause. Checked on the accepted hits only — a row that was
    # never the target has no reference obligation.
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
    """Score every query. `retriever` defaults to plain dense search.

    With `expected_pass` (a recorded baseline), the verdict is "no regressions":
    it fails only when a query the baseline passed now fails. Without it, the
    verdict is strict — every query must pass. See `compare_to_baseline`."""
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

    # How the first accepted hit of each pass qualified. Informative about a
    # single run, but NOT comparable between runs or retrievers: among tied
    # byte-identical rows, which kind ranks first is arbitrary (see
    # `_clause_rows`). Compare which queries passed, not these counts.
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
    """(regressions, improvements), each a sorted list of query ids.

    Why a baseline at all: the gate's honest score is 11-13 of 16, with every
    failure diagnosed as genuine ranking weakness. A strict gate is therefore red
    on every run, and a check that is always red cannot tell "still 11" from
    "dropped to 9" without someone reading the log. `pipeline.evaluate` passes
    at its baseline; this is the same contract.

    A new pass is reported but never fails the run, and it is never recorded
    automatically — blessing the current state is an explicit `--update-baseline`,
    so an improvement cannot quietly hide a regression elsewhere. Baseline ids
    no longer in the query set are ignored, so removing a query is not a
    regression (and should be a visible edit to the query set anyway).
    """
    current = {r.query_id for r in results}
    passing = {r.query_id for r in results if r.passed}
    regressions = sorted((expected_pass & current) - passing)
    improvements = sorted(passing - expected_pass)
    return regressions, improvements


def baseline_key(retriever: str, tokenizer: str, k: int) -> str:
    """One baseline per configuration. The tokenizer is part of the key only for
    arms that use it, so `dense` has one entry rather than three identical ones."""
    if retriever in ("bm25", "hybrid", "hybrid-brute"):
        return f"{retriever}/{tokenizer}/k={k}"
    return f"{retriever}/k={k}"


def load_baseline(path: Path, key: str, collection: str) -> set[str] | None:
    """The expected-pass set for `key`, or None when no baseline applies.

    A baseline recorded against a different collection does NOT apply: a
    different model, schema or index is a different system, and scoring it
    against another system's baseline would call a model swap a regression (or
    hide one). That case is logged loudly and falls back to the strict verdict.
    """
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

    settings = load_settings(require_api_key=needs_api_key(args.retriever))
    collection = open_collection(settings)
    embedder = Embedder(settings.api_key, settings.model, settings.request_delay)
    retriever = build_retriever(args.retriever, collection, embedder, args.pool, args.tokenizer)

    # The configuration is echoed in full because a score is meaningless without
    # it — these arms are meant to be compared, and a bare "11/16" in a log
    # would not say which one produced it.
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
        # The baseline records top-k passes, not ranks, so it cannot vouch for a
        # tighter rank limit.
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
