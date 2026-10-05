from __future__ import annotations

import argparse
import logging
import sys

from .chat import build_synthesizer
from .config import load_settings
from .embed import Embedder
from . import documents, references
from .lookup import has_answer, lookup, raw_document_provider, render, route
from .retrievers import build_retriever
from .store import (corpus_documents, describe_scope, document_resolver, open_collection,
                    resolve_scope)

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
        choices=("null", "ollama"),
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
    parser.add_argument(
        "--filter",
        help="With --list-documents: only those matching this name, number or party",
    )
    parser.add_argument(
        "--limit", type=int, default=25,
        help="With --list-documents: how many to print (default: 25)",
    )
    parser.add_argument(
        "--route",
        choices=("auto", "lookup", "search"),
        default="auto",
        help="auto answers core-field questions from the raw extraction and searches the rest (default)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show retrieval scores and ids")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.list_documents:
        settings = load_settings()
        collection = open_collection(settings)
        if args.filter:
            matches = document_resolver(collection, settings=settings)(args.filter, args.limit)
            print(f"{matches.total} document(s) matching {args.filter!r}")
            for document in matches.documents:
                print(f"  {document.document_key[:12]}  {document.describe()}")
            if matches.truncated:
                print(f"  ...and {matches.truncated} more")
            return 0
        available = corpus_documents(collection, settings=settings)
        print(f"{len(available)} document(s) in {settings.collection}")
        for key, name in list(available.items())[: args.limit]:
            print(f"  {key[:12]}  {name or '(name unknown — embedding views not on disk)'}")
        if len(available) > args.limit:
            print(f"  ...and {len(available) - args.limit} more (--limit, or --filter to narrow)")
        return 0

    if not args.question:
        parser.error("a question is required (or pass --list-documents)")

    settings = load_settings()
    collection = open_collection(settings)

    scope: dict[str, str] = {}
    question = args.question
    if args.document:
        scope = resolve_scope(args.document, collection, settings=settings)
    else:
        mention = documents.parse(question, document_resolver(collection, settings=settings))
        if mention.problem:
            print(f"{mention.problem}")
            return 1
        if mention.found:
            scope, question = mention.scope, mention.remainder
    if scope:
        print(f"scope: {describe_scope(scope)}\n")

    target = route(question) if args.route != "search" else None
    if args.route == "lookup" and target is None:
        parser.error("not a core-field question (nama/nomor kontrak, para pihak, tanggal, angka penting)")

    citation = references.parse(question) if target is None else None
    citation_only = citation is not None and not citation.remainder

    if target is not None:
        if not scope:
            available = corpus_documents(collection, settings=settings)
            if len(available) > 1:
                print(f"{target.label} berbeda untuk setiap kontrak — sebutkan dokumennya "
                      f"(nama file, nomor kontrak, atau penandatangan), atau gunakan --document. "
                      f"{len(available)} dokumen tersedia; --list-documents menampilkannya.")
                return 1
            scope = available
        answers = lookup(target, scope, raw_document_provider(settings))
        if has_answer(answers) or args.route == "lookup":
            print(render(target, answers))
            if args.verbose:
                print(f"\nroute: lookup ({target.field}{'/' + target.subtype if target.subtype else ''})")
            return 0
        print(f"({target.label.lower()} tidak ada di data inti — beralih ke pencarian klausul)\n")

    pinned = references.ReferenceResult()
    if citation is not None:
        pinned = references.find(citation, collection, set(scope) or None, args.k)
        if not pinned.found:
            print(f"({citation} tidak ditemukan sebagai label — hasil dari pencarian biasa)\n")
        elif pinned.part_relaxed:
            print(f"({citation} ditemukan di bagian lain dari yang disebutkan)\n")

    synthesizer = build_synthesizer(args.synthesizer, settings)
    hits = pinned.hits
    if not (citation_only and pinned.found):
        embedder = Embedder(settings.model, settings.host, num_gpu=settings.query_num_gpu)
        retriever = build_retriever(args.retriever, collection, embedder, args.pool, args.tokenizer)
        query = citation.remainder if citation is not None and citation.remainder else question
        searched = retriever.search(query, args.k, set(scope) or None)
        hits = references.merge(pinned.hits, searched, args.k)

    if not hits and scope:
        print(f"(nothing matched in {describe_scope(scope)} — try without --document)")

    scope_note = describe_scope(scope) if scope else ""

    try:
        answer = synthesizer.synthesize(question, hits, scope_note)
    except Exception as exc:
        logger.error("synthesis failed (%s: %s) — falling back to the retrieved clauses",
                     type(exc).__name__, exc)
        from .chat import NullSynthesizer

        answer = NullSynthesizer().synthesize(question, hits, scope_note)
        print(answer.text)
        print("\n(synthesis unavailable; the clauses above are the raw retrieval result)")
        return 1

    if args.verbose:
        print(f"retriever   : {args.retriever} (k={args.k})")
        print(f"scope       : {describe_scope(scope) if scope else 'whole corpus'}")
        if citation is not None:
            print(f"citation    : {citation} -> {len(pinned.hits)} pinned"
                  + (f" (tier {pinned.tier}" + (f", +{pinned.expanded} children" if pinned.expanded else "") + ")"
                     if pinned.found else " (none)")
                  + (f", search query {citation.remainder!r}" if citation.remainder else ", no search"))
        print(f"synthesizer : {args.synthesizer}" + (f" / {answer.model}" if answer.model else ""))
        print(f"hits        : {len(hits)} retrieved, {len(answer.sources)} after collapsing duplicates")
        for hit in hits:
            print(f"  [{hit.score:.3f}] {hit.id} {' '.join((hit.text or '').split())[:70]}")
        print()

    print(answer.text)

    if args.synthesizer != "null" and answer.sources:
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
