"""Ask a question against the contract corpus: retrieve, then optionally answer.

The ONLY place the retrieval layer and the synthesis layer meet. Both sides are
chosen by flag and neither knows about the other — `--retriever` picks how
clauses are found, `--synthesizer` picks what happens to them afterwards. That
separation is the point: swapping either is a flag, not an edit.

    python -m retrieval.ask "kewajiban penyedia mengasuransikan pekerjaan"
    python -m retrieval.ask "berapa denda keterlambatan?" --synthesizer mistral
    python -m retrieval.ask "ruang lingkup pekerjaan" --retriever hybrid -k 8

Default is `--synthesizer null`: retrieval output, no model call, no tokens.
Synthesis is opt-in because the honest default for a legal corpus is the source
text, and because a generated paragraph is the one part of this system that
cannot be checked against ground truth.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .chat import build_synthesizer
from .config import load_settings, needs_api_key
from .embed import Embedder
from .retrievers import build_retriever
from .store import open_collection

logger = logging.getLogger("retrieval.ask")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Windows consoles default to cp1252, which cannot encode characters a chat
    # model routinely emits — an emoji in a reply crashed this command with a
    # UnicodeEncodeError before the answer was printed at all. The contract text
    # itself is also full of typographic quotes and dashes. Replace rather than
    # raise: a mangled character is a cosmetic problem, a traceback loses the
    # whole answer. Verify console characters against the actual bytes;
    # never trust what the terminal renders.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Query the contract corpus")
    parser.add_argument("question")
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
    parser.add_argument("--verbose", action="store_true", help="Show retrieval scores and ids")
    args = parser.parse_args()

    settings = load_settings(require_api_key=needs_api_key(args.retriever, args.synthesizer))
    collection = open_collection(settings)
    embedder = Embedder(settings.api_key, settings.model, settings.request_delay)
    retriever = build_retriever(args.retriever, collection, embedder, args.pool, args.tokenizer)
    synthesizer = build_synthesizer(args.synthesizer, settings)

    hits = retriever.search(args.question, args.k)

    try:
        answer = synthesizer.synthesize(args.question, hits)
    except Exception as exc:
        # Synthesis is the one part of this command that depends on a live
        # third-party endpoint, and a rate limit or outage there is expected
        # rather than exceptional. Retrieval already succeeded, so fall back to
        # showing the clauses instead of losing that work to a traceback.
        logger.error("synthesis failed (%s: %s) — falling back to the retrieved clauses",
                     type(exc).__name__, exc)
        from .chat import NullSynthesizer

        answer = NullSynthesizer().synthesize(args.question, hits)
        print(answer.text)
        print("\n(synthesis unavailable; the clauses above are the raw retrieval result)")
        return 1

    if args.verbose:
        print(f"retriever   : {args.retriever} (k={args.k})")
        print(f"synthesizer : {args.synthesizer}" + (f" / {answer.model}" if answer.model else ""))
        print(f"hits        : {len(hits)} retrieved, {len(answer.sources)} after collapsing duplicates")
        for hit in hits:
            print(f"  [{hit.score:.3f}] {hit.id} {' '.join((hit.text or '').split())[:70]}")
        print()

    print(answer.text)

    if args.synthesizer != "null" and answer.sources:
        # Printed even when the model cited nothing, so a reader can always
        # check the answer against the clauses it was given.
        print("\nSumber:")
        for n, source in enumerate(answer.sources, start=1):
            copies = f" (x{source.copies} identik)" if source.copies > 1 else ""
            print(f"  [{n}] {source.citation} — {source.sub_document}{copies}")
        if answer.usage_tokens:
            print(f"\n({answer.usage_tokens} tokens)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
