# Architecture — what each file does

Map of the codebase from PDF input through to keyword output. Brief by design;
for command syntax see `README.md`.

---

## The shape of the system

Two extraction front ends feed one shared back end, and a post-processing step
reduces the result to keywords.

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
                             raw_extraction.json
                                       │
                          keywords/clean_json.py
                                       │
                            clean_extraction.json
```

**The key idea:** `PageProbe` (a page's words, each with a bounding box) is the
only thing the shared stages know about. They never touch a PDF. So the OCR
pipeline reimplements just the two PDF-specific pieces and reuses everything
downstream — which is why both produce the same JSON schema by construction
rather than by convention.

---

## Entry points

| File | Role |
|---|---|
| `pipeline/main.py` | Native pipeline CLI. Orchestrates stages 1–9 for a born-digital PDF and writes `raw_extraction.json`. |
| `pipeline/ocr_main.py` | OCR pipeline CLI. Same output, same stages, but words come from Tesseract instead of pdfplumber. Imports the shared stages; modifies nothing. |
| `keywords/clean_json.py` | Reduces either pipeline's output to `clean_extraction.json`. |

---

## Shared stages (used by both pipelines)

| File | Stage | What it does |
|---|---|---|
| `probe.py` | 1 | Reads the PDF into `PageProbe` — per-page size, fonts, ruling lines, and every word with its box. Native only; the OCR pipeline builds `PageProbe` itself. |
| `router.py` | 2 | Decides per page whether text is native, needs OCR, or is ambiguous. Used by the native pipeline to flag gaps. |
| `profiles.py` | 3 | Scores the document against `profiles/*.json` and picks one. A profile supplies sub-document markers and validation invariants; it does not drive numbering/depth parsing, but its sub-document markers DO decide one node-type classification in `tree.py` (see below) — not purely cosmetic labeling. |
| `layout.py` | 4 | Classifies each page as `single_column` / `two_column` / `ruled_table` / `form` / `mixed` / `blank`, from geometry alone. Also computes where the right column starts. |
| `blocks.py` | 5 | Turns words into ordered text blocks, splitting two-column rows and sorting into true reading order. Also extracts ruled tables via pdfplumber (native only). |
| `numbering.py` | — | Recognizes numbering tokens (`3.`, `21.4`, `a.`, `BAB II`, …) in a style-agnostic way. Used by `tree.py` to decide where nodes begin. |
| `tree.py` | 6 | Builds the recursive node tree — clauses, subclauses, list items, headings — from the flat blocks. The most intricate file in the project. A `decimal_plain` numbering is classified `clause` only on pages inside the profile's clause-bearing sub-document (`expected_invariants.clause_sequence_scope`), computed by `main.py`/`ocr_main.py` and passed in — otherwise it's a plain `list_item`. Requires sub-document assignment (`main.assign_sub_documents`) to run BEFORE `build_tree`, not after. |
| `entities.py` | 7 | Tags entities, resolves cross-references between nodes, marks modality. |
| `core_fields.py` | 8 | Resolves the six guaranteed core fields (document type, name, number, parties, dates, numbers) by regex/heuristic cascade. Never guesses — unresolved means `null`. |
| `validate.py` | 9 | Runs generic quality checks and embeds a `quality` block in the output. A hard failure flips `pipeline_status` but still writes the file. |
| `normalize.py` | — | Indonesian currency, date, number-word and rate parsing. |
| `schema.py` | — | The shared value-object shape (`{value, raw, confidence, method, …}`) and ID generation. |

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
| `keywords/clean_json.py` | Flattens core value-objects to plain values, attaches the keyword `body`, and writes `clean_extraction.json`. |

Keywords are mined from the **node tree**, not from page text — a node is a
semantic unit while a page is an arbitrary rectangle, and mining page text lets
phrases run across line breaks.

---

## Supporting files

| File | Role |
|---|---|
| `pipeline/evaluate.py` | Scores output against ground truth, plus a permanent regression checklist of every bug ever fixed. |
| `pipeline/sample_review.py` | Builds a stratified CSV sample for human review. |
| `profiles/*.json` | Document-family profiles. `generic_contract_v1` is the mandatory fallback. |
| `ground_truth/*.json` | Hand-verified expectations and `regression_checks.json`. |
| `requirements.txt` | `pdfplumber` for the native path; `pytesseract`/`PyMuPDF`/`opencv-python`/`numpy` for OCR; `yake` for the optional YAKE keyword backend (RAKE, the default, is hand-implemented and needs nothing extra). Tesseract's own binary and `ind` language data are not pip-installable. |

---

## Outputs

| File | Contents |
|---|---|
| `raw_extraction.json` | Full fidelity — every node, page, table, entity, and the quality block. The audit artifact. Kept. |
| `clean_extraction.json` | Flattened core fields plus a keyword `body`. Roughly 0.7% the size. What downstream storage and search consume. |

Both CLIs (`pipeline.main`, `pipeline.ocr_main`) write these two files flat
into whatever `--out` directory is given — there is no built-in per-document
subfolder scheme. `output/` and `output_ocr/` (both gitignored) are where
local runs land; when multiple PDFs are run into one folder, files are
organized by hand into `raw/`, `clean/`, `log/` subfolders (one file per
document, named after the source PDF) to avoid collisions — see `README.md`'s
Layout section.

---

## Two rules worth preserving

1. **The OCR pipeline never edits the shared stages.** When OCR output doesn't
   fit, convert the OCR output to match what those stages already expect — the
   way row-top normalization and pixel→point scaling do — rather than loosening
   a shared assumption.
2. **Every fixed bug gets a `regression_checks.json` entry.** The random review
   sample changes between runs and isn't comparable round to round; the
   checklist is the same check every time.
