from __future__ import annotations

import argparse
import logging
import sys

from .chat import build_synthesizer
from .config import load_settings, needs_api_key
from .embed import Embedder
from .retrievers import build_retriever
from .store import corpus_documents, describe_scope, open_collection, resolve_scope

logger = logging.getLogger("retrieval.ask")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Query the contract corpus")
    parser.add_argument("question", nargs="?")
    parser.add_argument("-k", type=int, default=5, help="Clauses to retrieve (default: 5)")
    parser.add_argument(
        "--retriever",
        choices=("dense", "brute", "bm25", "hybrid", "hybrid-brute"),
        default="hybrid",
        help="How to find clauses (default: hybrid — the best-measured arm on this corpus)",
    )
    parser.add_argument("--tokenizer", choices=("plain", "nostop", "stem"), default="plain")
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument(
        "--synthesizer",
        choices=("null", "mistral"),
        default="null",
        help="null prints the retrieved clauses and makes no model call (default)",
    )
    parser.add_argument(
        "--document",
        help="Restrict the search to one contract: a filename substring, a document_key "
        "prefix, or a comma-separated list. Default: the whole corpus",
    )
    parser.add_argument(
        "--list-documents",
        action="store_true",
        help="Print the documents in the collection and exit",
    )
    parser.add_argument("--verbose", action="store_true", help="Show retrieval scores and ids")
    args = parser.parse_args()

    # Listing needs no API key, so it runs before anything that demands one.
    if args.list_documents:
        settings = load_settings(require_api_key=False)
        for key, name in corpus_documents(open_collection(settings)).items():
            print(f"  {key[:12]}  {name or '(name unknown — embedding views not on disk)'}")
        return 0

    if not args.question:
        parser.error("a question is required (or pass --list-documents)")

    settings = load_settings(require_api_key=needs_api_key(args.retriever, args.synthesizer))
    collection = open_collection(settings)

    scope: dict[str, str] = {}
    if args.document:
        scope = resolve_scope(args.document, collection)
        # Unconditional, not just under --verbose: a scoped answer that looks
        # corpus-wide is this flag's dangerous failure mode.
        print(f"scope: {describe_scope(scope)}\n")

    embedder = Embedder(settings.api_key, settings.model, settings.request_delay)
    retriever = build_retriever(args.retriever, collection, embedder, args.pool, args.tokenizer)
    synthesizer = build_synthesizer(args.synthesizer, settings)

    hits = retriever.search(args.question, args.k, set(scope) or None)
    if not hits and scope:
        # Distinct from a corpus-wide miss: the cause is the scope, not the question.
        print(f"(nothing matched in {describe_scope(scope)} — try without --document)")

    scope_note = describe_scope(scope) if scope else ""

    try:
        answer = synthesizer.synthesize(args.question, hits, scope_note)
    except Exception as exc:
        # Retrieval already succeeded, so a third-party outage falls back to
        # showing the clauses rather than losing that work to a traceback.
        logger.error("synthesis failed (%s: %s) — falling back to the retrieved clauses",
                     type(exc).__name__, exc)
        from .chat import NullSynthesizer

        answer = NullSynthesizer().synthesize(args.question, hits, scope_note)
        print(answer.text)
        print("\n(synthesis unavailable; the clauses above are the raw retrieval result)")
        return 1

    if args.verbose:
        print(f"retriever   : {args.retriever} (k={args.k})")
        print(f"scope       : {describe_scope(scope) if scope else 'whole corpus'}")
        print(f"synthesizer : {args.synthesizer}" + (f" / {answer.model}" if answer.model else ""))
        print(f"hits        : {len(hits)} retrieved, {len(answer.sources)} after collapsing duplicates")
        for hit in hits:
            print(f"  [{hit.score:.3f}] {hit.id} {' '.join((hit.text or '').split())[:70]}")
        print()

    print(answer.text)

    if args.synthesizer != "null" and answer.sources:
        # Printed even when the model cited nothing, so the answer can always
        # be checked against the clauses it was given.
        print("\nSumber:")
        for n, source in enumerate(answer.sources, start=1):
            copies = f" (x{source.copies} identik)" if source.copies > 1 else ""
            where = (
                "" if source.citation.startswith(source.sub_document_label)
                else f" — {source.sub_document_label}"
            )
            print(f"  [{n}] {source.citation}{where}{copies}")
        if answer.usage_tokens:
            print(f"\n({answer.usage_tokens} tokens)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
