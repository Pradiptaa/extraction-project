# Architecture — what each file does

Map of the codebase from PDF input through to keyword output and retrieval.
Brief by design; for command syntax see `README.md`.

---

## The shape of the system

Two extraction front ends feed one shared back end. The raw result then forks
into two independent derived views — keywords, and retrieval.

```
                 native PDF                     scanned PDF
                      │                              │
              pipeline/main.py              pipeline/ocr_main.py
              (probe + tables)              (render, OCR, tables)
                      │                              │
                      └──────────► PageProbe ◄───────┘
                                       │
                    shared stages: layout → blocks → tree →
                    entities → core_fields → validate
                                       │
                          <pdf-stem>_raw.json
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
      keywords/clean_json.py                retrieval/build_embedding_view.py
                 │                                           │
      <pdf-stem>_cleaned.json              <pdf-stem>_embedding_view.json
                                             (1 row per tree node + 1 per table row)
                                                             │
                                              retrieval/load.py → Mistral
                                                             │
                                                   ChromaDB collection
                                                             │
                                          retrievers.py (dense/bm25/hybrid)
                                                             │
                                    ┌────────────────────────┴────────┐
                                    │                                 │
                        retrieval_evaluate.py                  ask.py → chat.py
                          (the regression gate)            (answer synthesis)
```

The two forks never meet: keywords do not feed retrieval, and retrieval does
not read the cleaned file. Both start from `<pdf-stem>_raw.json`, which is the
only artifact either depends on.

**The key idea:** `PageProbe` (a page's words, each with a bounding box) is the
only thing the shared stages know about. They never touch a PDF. So the OCR
pipeline reimplements just the two PDF-specific pieces and reuses everything
downstream — which is why both produce the same JSON schema by construction
rather than by convention.

---

## Entry points

| File | Role |
|---|---|
| `pipeline/main.py` | Native pipeline CLI. Orchestrates stages 1–9 for a born-digital PDF and writes `<pdf-stem>_raw.json`. |
| `pipeline/ocr_main.py` | OCR pipeline CLI. Same output, same stages, but words come from Tesseract instead of pdfplumber. Imports the shared stages; modifies nothing. |
| `keywords/clean_json.py` | Reduces either pipeline's output to `<pdf-stem>_cleaned.json` (`--method yake` -> `<pdf-stem>_cleaned_yake.json`). |
| `retrieval/build_embedding_view.py` | Projects a raw file into `<pdf-stem>_embedding_view.json`: one row per tree node and one per ruled-table row. |
| `retrieval/load.py` | Embeds views and loads them into Chroma. Resumable; `--dry-run` costs nothing; `--reuse-from` copies stored vectors instead of re-embedding. |
| `retrieval/reindex.py` | Rebuilds a collection's HNSW index from stored vectors. No embedding calls. |
| `retrieval/retrieval_evaluate.py` | The retrieval regression gate, judged against a recorded baseline. `--retriever hybrid\|dense\|bm25\|brute\|hybrid-brute`. |
| `retrieval/ask.py` | Ask a question: retrieve, then optionally synthesize an answer. |

Only `pipeline.main` / `pipeline.ocr_main` need a PDF. Everything after them
works from JSON, so the retrieval stage can be re-run without re-extracting and
the gate can be re-run without re-embedding.

---

## Shared stages (used by both pipelines)

| File | Stage | What it does |
|---|---|---|
| `probe.py` | 1 | Reads the PDF into `PageProbe` — per-page size, fonts, ruling lines, and every word with its box. Native only; the OCR pipeline builds `PageProbe` itself. |
| `router.py` | 2 | Decides per page whether text is native, needs OCR, or is ambiguous. Used by the native pipeline to flag gaps. |
| `profiles.py` | 3 | Scores the document against `profiles/*.json` and picks one. A profile supplies sub-document markers and validation invariants; it does not drive numbering/depth parsing, but its sub-document markers DO decide one node-type classification in `tree.py` (see below) — not purely cosmetic labeling. |
| `layout.py` | 4 | Classifies each page as `single_column` / `two_column` / `ruled_table` / `form` / `mixed` / `blank`, from geometry alone — a histogram of each line's **leftmost** word x0. Also computes where the right column starts. Leftmost, not first-in-sort-order: a left-column clause number often sits a fraction of a point lower than the right-column text beside it, and taking the first word of a (top, x0) sort made the left column invisible, dropped the page to `single_column`, and folded each clause's first sub-clause into its title (`bug_024`, 23 clauses in 4 of 6 specimens). |
| `blocks.py` | 5 | Turns words into ordered text blocks, splitting two-column rows and sorting into true reading order. Also extracts ruled tables via pdfplumber (native only). |
| `numbering.py` | — | Recognizes numbering tokens (`3.`, `21.4`, `a.`, `BAB II`, …) in a style-agnostic way. Used by `tree.py` to decide where nodes begin. |
| `tree.py` | 6 | Builds the recursive node tree — clauses, subclauses, list items, headings — from the flat blocks. The most intricate file in the project. A `decimal_plain` numbering is classified `clause` only on pages inside the profile's clause-bearing sub-document (`expected_invariants.clause_sequence_scope`), computed by `main.py`/`ocr_main.py` and passed in — otherwise it's a plain `list_item`. Requires sub-document assignment (`main.assign_sub_documents`) to run BEFORE `build_tree`, not after. |
| `entities.py` | 7 | Tags entities, resolves cross-references between nodes, marks modality. |
| `core_fields.py` | 8 | Resolves the six guaranteed core fields (document type, name, number, parties, dates, numbers) by regex/heuristic cascade. Never guesses — unresolved means `null`. |
| `validate.py` | 9 | Runs generic quality checks and embeds a `quality` block in the output. A hard failure flips `pipeline_status` but still writes the file. |
| `normalize.py` | — | Indonesian currency, date, number-word and rate parsing. |
| `schema.py` | — | The shared value-object shape (`{value, raw, confidence, method, …}`) and ID generation. `node_id` is a plain sequential counter (`n_0001`, `n_0002`, ...) assigned in tree-build order — not derived from content. It is stable across repeated runs on the same (code, input) pair, because every upstream stage sorts explicitly by geometry before consuming (no dict/set iteration order anywhere in the path) — verified across 3 independent runs, pinned by two `node_id_equals` regression checks. It is NOT a durable cross-run identity, though: it's positional, so any upstream change that adds/removes a node shifts every later node_id even when that node's own content is unchanged. Anything needing a durable key across pipeline versions must derive it from content instead — see `retrieval/schema.py`'s `embedding_id`, which `hierarchy_path + label_normalized` alone turned out to be far from enough for (it collided on 48.6% of the corpus). |

---

## OCR-specific parts (all inside `pipeline/ocr_main.py`)

Everything below exists only to produce a `PageProbe` that behaves exactly like
a native one.

| Function | What it does |
|---|---|
| `_render_gray` | Renders a page to a grayscale image at the chosen DPI (PyMuPDF). |
| `_estimate_skew` / `_deskew` | Finds page tilt by projection profile — rotating through candidate angles and picking the one where text baselines are most level — then corrects it. Straight pages correctly get 0°. |
| `_binarize` | Otsu threshold, so recognition sees clean black-on-white. |
| `_ocr_words` | Runs Tesseract and converts its **pixel** boxes into **PDF points**. Critical: every downstream threshold is calibrated in points. |
| `_ocr_words_two_pass` / `_reconcile` | Runs two page-segmentation modes and merges them. PSM 4 gives good word segmentation; PSM 11 finds tokens PSM 4 drops. `_reconcile` decides conflicts — pass 2 may fix punctuation inside a token but never change its letters or strip a trailing period. |
| `_normalize_row_tops` | Gives every word on a visual row the same top/bottom. pdfplumber reports line-box tops (uniform per row); Tesseract reports ink tops (varying by capitals and ascenders). Downstream code assumes the former. |
| `_detect_rule_segments` / `_detect_rule_grid` | Finds table rules by morphological opening and qualifies them into a real grid: a horizontal rule must be long, a vertical rule must cross at least two of them, and the grid must enclose at least two cells. This is what keeps a letterhead emblem from being read as a table. |
| `extract_table_blocks_ocr` | Builds table cell grids from the qualified rules, filling cells with the words inside them. Replaces pdfplumber's vector-line table extraction. |
| `_neutralize_oracle_check` | Marks the `dual_parser_oracle` validation check as skipped — it compares against a native text layer, which a scan doesn't have. |

---

## Keyword extraction

| File | What it does |
|---|---|
| `keywords/stopwords_id.txt` | 757 Indonesian stopwords, verbatim from stopwords-iso. Not edited directly — adjustments live in code. |
| `keywords/extractor.py` | Loads stopwords (rescuing `pihak`/`waktu`/`bagian`, adding contract boilerplate and structural words), seeds guaranteed terms from the already-resolved core fields, then mines the rest with RAKE (default) or YAKE. Filters nonsense n-grams and near-duplicates. |
| `keywords/clean_json.py` | Flattens core value-objects to plain values, attaches the keyword `body`, and writes `<pdf-stem>_cleaned.json`. |

Keywords are mined from the **node tree**, not from page text — a node is a
semantic unit while a page is an arbitrary rectangle, and mining page text lets
phrases run across line breaks.

---

## Retrieval (`retrieval/`)

A separate stage that reads `<pdf-stem>_raw.json` and ends in a queryable vector
collection. It never imports from `pipeline/` and `pipeline/` never imports from
it.

| File | What it does |
|---|---|
| `__init__.py` | Turns Chroma's anonymous telemetry off at package import. chromadb defaults it on and reads it when a client is created, so setting it in `.env` covered only commands that loaded settings first — not the tests or any script that skipped that. It has to be process-wide: Chroma refuses two clients on one path with different settings. |
| `schema.py` | `EMBEDDING_SCHEMA_VERSION` (currently 2.1.0) and `embedding_id()`. Mirrors `pipeline/schema.py`'s role. The id is a hash of `document_key + sub_document + page + path + label + text + occurrence` — deliberately **not** `node_id`, which is positional and would orphan vectors on any upstream insertion. Every component was added because measurement showed the previous key collapsing rows on upsert. Because `text` is in the hash, an unchanged id guarantees unchanged text — which is what makes vector reuse across schema versions safe. |
| `build_embedding_view.py` | Projects a raw file 1:1 into `<pdf-stem>_embedding_view.json`: one row per `structure[]` node and, since 2.1.0, one per `tables[]` row (`node_type: table_row`, path `[table_id, row]`, cells joined by ` \| `, cross-references carried as `refs`). **Not the chunker** — no splitting or merging, so correctness is checkable by row-count parity, reported per source. Raises rather than emitting a view with duplicate ids. |
| `config.py` | Settings from `.env`, plus the naming rules: `collection_name()` encodes model + schema + index tag, and `INDEX_METADATA` pins the HNSW parameters. Also decides when a key is required (`needs_api_key` — bm25 with no synthesis needs none) and resolves a relative `CHROMA_DB_PATH` from `JSONextraction/`, never the current directory. `Settings` keeps the key out of its `repr`. |
| `store.py` | Opens the configured collection for reading, with a distinct error for "no store", "no such collection" and "empty". Shared by the gate and `ask` so neither command imports the other. |
| `embed.py` | Mistral embedding calls. Retry/backoff on 429/5xx and timeouts only; a 401 or 422 fails immediately. Enforces two invariants: every vector has the width of the first one seen, and a response never has fewer vectors than inputs. |
| `load.py` | The resumable batch job: embed → upsert → append to a JSONL manifest, in that order. **Chroma alone decides what is done** — the manifest is an audit log; when it claims rows the collection does not hold (a deleted and recreated collection), those rows are embedded again. A transient failure is logged and skipped; any other failure stops the run. `--reuse-from` copies vectors by id from a collection built by the same model, refusing any id whose stored text differs. |
| `reindex.py` | Rebuilds a collection's HNSW index from vectors already stored — no embedding calls. Exists because Chroma fixes index parameters at creation, so changing them means a new collection. `--verify` checks recall against an exact brute-force scan. |
| `retrievers.py` | `DenseRetriever`, `Bm25Retriever`, `HybridRetriever` (RRF) behind one `Hit`-returning interface, plus the tokenizers. `BruteForceRetriever` is an exact-search **reference ceiling**, not a production strategy: comparing it against `dense` is how you tell index loss apart from genuine ranking weakness. Adding a strategy means adding a class here, not touching the harness. |
| `retrieval_evaluate.py` | The regression gate. Same shape as `pipeline/evaluate.py` — per-check PASS/FAIL and a summary — but the exit code is judged against `ground_truth/retrieval_baseline.json`: exit 1 only when a query the baseline passes now fails. A baseline recorded on a different collection does not apply. A query targets a **clause** (`hierarchy_path`) or a **row by content** (`text_contains`, for table rows), optionally narrowed by `node_type`, an exact substring, and `expect_ref` — the hit's cross-reference must resolve to a given clause. |
| `chat.py` | Answer synthesis over retrieved clauses — a layer **above** retrieval. `Synthesizer` is a `Protocol`; `NullSynthesizer` (default) makes no model call, `MistralSynthesizer` does, with an extractive prompt that forbids outside knowledge and guessing. Collapses duplicate clauses before prompting. |
| `ask.py` | The only place retrieval and synthesis meet. `--retriever` picks how clauses are found, `--synthesizer` what happens next; neither side knows about the other. |

**The chat layer is one-directional and it is enforced.** `chat.py` imports from
the retrieval path; nothing in the retrieval path imports `chat.py`, asserted by
a test that parses every module. This is the same rule `ocr_main.py` follows
around the shared stages, and it is what lets synthesis be swapped or deleted
without touching retrieval — including running the gate with no chat model
configured. The gate deliberately never scores generated prose: whether the
right clause came back is checkable against ground truth, whether the paragraph
reads well is not.

Three things here are counter-intuitive enough to be worth knowing before
changing anything:

- **Measure index recall by distance, not by id.** 60% of rows are duplicate
  text, so an id comparison largely measures arbitrary tie-breaking between
  identical rows rather than whether the index found the right content.
- **Asking Chroma for more results is not a quality knob.** A larger
  `n_results` makes HNSW explore differently and can return a *worse* top-k.
  Index quality is set by `INDEX_METADATA`; pool size is only a fusion input.
- **The gate accepts three kinds of hit** — `exact`, `descendant` and
  `equivalent` (byte-identical text under a different clause key). The third
  exists because without it the score swings by 4 of 16 queries on tie-ordering
  alone. It is bounded: text under 60 characters qualifies only if it also sits
  under the target clause once the section letter is stripped, so a generic
  fragment ("Pengadilan.") cannot pass for an unrelated clause. The pass/fail
  verdict is tie-stable; the *kind* of a pass is not (`dense` and `brute` pass
  the same queries with different kind counts), so compare pass sets, not
  kind counts.
- **A gate that is red at baseline cannot gate.** The honest score is below
  100%, with every failure diagnosed, so a strict all-must-pass verdict failed
  every run and could not tell "still 17" from "dropped to 15". The baseline
  file is what makes the exit code mean "no regression". Recording a new
  baseline is an explicit act that must come with an explanation — an
  improvement must not be allowed to hide a regression elsewhere.
- **Adding rows changes lexical scores for rows you did not touch.** Loading
  the 478 table rows cost `bm25` one query (q08) without any table row entering
  its top 8: more rows shift BM25's term weights and average length. Any corpus
  change needs every retriever arm re-scored, not only the one it was aimed at.

---

## Supporting files

| File | Role |
|---|---|
| `pipeline/evaluate.py` | Scores output against ground truth, plus a permanent regression checklist of every bug ever fixed. Checklist entries are `node`, `count`, or `table_refs` — the last checks cross-references out of ruled-table rows, which live outside the node tree (it runs the SSKK → SSUK case, `ns_12_sskk_keyed_row`). |
| `pipeline/sample_review.py` | Builds a stratified CSV sample for human review. |
| `profiles/*.json` | Document-family profiles. `generic_contract_v1` is the mandatory fallback. |
| `ground_truth/*.json` | Hand-verified expectations (one per specimen), `regression_checks.json` (node, count and `table_refs` checks), `retrieval_queries.json` — the query set the retrieval gate scores against — and `retrieval_baseline.json`, its recorded expected passes per retriever/tokenizer/k. |
| `retrieval/README.md` | How to run the retrieval stage: setup, load, the gate, expected scores, limitations. |
| `retrieval/tests/` | 138 unit tests for the retrieval and chat layers. Every embedder and chat client is faked, so the whole suite runs with **no API key and no tokens** — the scoring rules must be testable without depending on what a model says today. Also pins the secrets rules: `.env` and `chroma_data/` gitignored at any depth, no key in the template, none in `Settings`' repr. |
| `requirements.txt` | `pdfplumber` for the native path; `pytesseract`/`PyMuPDF`/`opencv-python`/`numpy` for OCR; `yake` for the optional YAKE keyword backend (RAKE, the default, is hand-implemented and needs nothing extra). Tesseract's own binary and `ind` language data are not pip-installable. |
| `requirements-retrieval.txt` | Kept separate so the extraction pipeline has no dependency on the retrieval stack: `chromadb`, `mistralai`, `tenacity`, `python-dotenv`, `rank-bm25`, `Sastrawi`, `numpy`. |
| `retrieval/.env` | API key and pinned model names. Gitignored (as is any `.env`, at any depth); `.env.example` is the only tracked env file. |

---

## Outputs

| File | Contents |
|---|---|
| `<pdf-stem>_raw.json` | Full fidelity — every node, page, table, entity, and the quality block. The audit artifact. Kept. |
| `<pdf-stem>_cleaned.json` | Flattened core fields plus a keyword `body`. Roughly 0.7% the size. What downstream storage and search consume. |
| `<pdf-stem>_embedding_view.json` | One row per tree node and per non-blank table row: a durable `embedding_id`, its place in the document, and the text to embed. A 1:1 projection, not chunks. |
| `chroma_data/` | The persistent Chroma store plus the loader's JSONL audit manifest. Gitignored, regenerable, and **not** a source of truth — it can be rebuilt from the embedding views. |

Both CLIs (`pipeline.main`, `pipeline.ocr_main`) write these two files flat
into whatever `--out` directory is given — there is no built-in per-document
subfolder scheme. `output/` and `output_ocr/` (both gitignored) are where
local runs land; when multiple PDFs are run into one folder, files are
organized by hand into `raw/`, `clean/`, `log/` subfolders (one file per
document, named after the source PDF) to avoid collisions — see `README.md`'s
Layout section.

---

## Five rules worth preserving

1. **The OCR pipeline never edits the shared stages.** When OCR output doesn't
   fit, convert the OCR output to match what those stages already expect — the
   way row-top normalization and pixel→point scaling do — rather than loosening
   a shared assumption.
2. **Every fixed bug gets a `regression_checks.json` entry.** The random review
   sample changes between runs and isn't comparable round to round; the
   checklist is the same check every time.
3. **The chat layer is one-directional.** `chat.py` imports from the retrieval
   path; nothing in the retrieval path imports `chat.py`, and a test enforces
   it. This is rule 1 applied a layer up — it is what lets synthesis be
   swapped, replaced or deleted without touching retrieval, and what lets the
   gate run with no chat model configured.
4. **The gate never scores generated prose.** Whether the right clause was
   retrieved is checkable against ground truth; whether a model wrote a good
   paragraph from it is not. Keeping synthesis outside
   `retrieval_evaluate.py` is what stops a regression gate from decaying into
   a vibe check.
5. **Change the system, never the expectations.** Ground truth — the
   per-specimen files and the retrieval query set — is built by reading the
   source, never by blessing pipeline output. When a score moves, explain which
   change moved it before recording a new baseline. Where the source itself is
   inconsistent (a PDF numbering its SSUK `1.119`, an SSKK citing a clause that
   does not exist), leave the value unresolved rather than guess.
