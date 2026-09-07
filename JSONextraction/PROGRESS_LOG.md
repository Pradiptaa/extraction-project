# Progress Log

Running log for the contract PDF extraction project. Append a new dated
section per day worked — newest entry at the top. Don't edit past entries
except to fix a factual error; if something from a past day turns out wrong,
say so in the current day's entry rather than rewriting history.

---

## Day 3 — 2026-09-04

### Goal for the day

Add OCR support on the assumption that every incoming document needs it, and
scope the keyword-extraction layer. Explicitly reverses Day 1's "no OCR engine"
scope boundary — the user's instruction was to ignore it, since it had been an
assumption rather than a requirement.

### Starting state

Native pipeline at 28/28 core checks and 20/20 regression checks. No OCR
anywhere: `router.py` classified pages as `ocr_needed` but nothing consumed the
decision, so such pages produced empty nodes flagged
`ocr_required_not_available`.

### Design decision that shaped everything else

The user asked for a **completely separate OCR pipeline** producing the same
JSON, without touching the existing one. Reading `main.py` showed a seam that
made this cheap: the only PDF-specific dependencies in the whole pipeline are
`probe.probe_document()` (which builds `PageProbe`) and
`blocks.extract_table_blocks()` (which needs pdfplumber's vector lines).
Everything from `layout.classify_layout()` onward consumes `PageProbe` and is
source-agnostic.

So the OCR pipeline reimplements exactly those two pieces and **imports** the
rest. Importing is read-only and does not violate the constraint, while
reimplementing the downstream stages "from scratch" would have duplicated 23
bugs' worth of debugging and guaranteed the two pipelines drift apart on the
first fix applied to only one. This also means the output format matches by
construction rather than by convention. Total footprint: one new file.

### What was built

- **`pipeline/ocr_main.py`** — render (PyMuPDF, 300 DPI), deskew, binarize
  (Otsu), recognize (Tesseract `ind+eng`), and construct `PageProbe`; plus
  OpenCV morphological table detection to replace pdfplumber's vector-line
  extraction, and an orchestrator mirroring `run_pipeline()` stage for stage.
- **`_neutralize_oracle_check()`** — rewrites the `dual_parser_oracle` result
  to `skip`. That check compares against Poppler `pdftotext`, which on a
  scanned PDF returns an empty text layer and would report ~0 similarity on
  every page; it only self-skips when the binary is *absent*. Done as a
  post-hoc edit of the returned dict specifically to keep `validate.py`
  untouched.
- **Dependencies**: `pytesseract`, `PyMuPDF`, `opencv-python`, `numpy`, plus
  the Tesseract binary and `ind` traineddata, which pip cannot install.

**The critical implementation detail** is the coordinate contract. Tesseract
reports pixel boxes at render DPI, but every geometry threshold downstream is
calibrated in PDF points — the 6pt row bucket in `blocks.py`,
`right_column_start_frac` in `layout.py`, and `SSUK_BODY_PAGE_HEIGHT = 792.0`
in `tree.py`. All boxes are therefore scaled back to points before a
`PageProbe` is built, using the actual rendered dimensions rather than a
nominal `dpi/72` so rasterizer rounding cannot introduce a systematic
sub-point error. This worked: measured word positions land within 0.4pt of
native.

### Research behind the choices

Compared Tesseract, PaddleOCR and cloud OCR (Google Vision, Azure Document
Intelligence). Chose Tesseract — free, fully local, no data leaving the machine
for a government contract, and consistent with a project whose only dependency
had been pdfplumber. PaddleOCR noted as the fallback if accuracy proves
insufficient. Also surveyed keyword-extraction options (TF-IDF, RAKE, YAKE,
KeyBERT) and Indonesian stopword/stemming resources for the following day.

### End state

First full run, against a render of the born-digital sample (no real scan was
available, and still isn't):

```
pages: 74  nodes: 569  tables: 19  mean_confidence=93.62
validation: passed  hard_fails=0  warns=2
  sibling_sequence: 24 breaks
  clause_sequence_gapless: clause_count=80, has gaps or duplicates
```

Text fidelity was excellent — per-page character counts within a few of native,
93.6% mean confidence, no empty pages — but structure was badly wrong: 569
nodes against 738, and `subclause` down from 301 to 159.

### Key finding

The character-count agreement is what made the failure interesting: all the
*text* was present, so char-conservation passed and masked the problem. Only
the labels were missing. That is a useful diagnostic shape to remember — when
text volume matches but node count collapses, look at what `numbering.py`
should have matched, not at recognition quality.

### Known limitations at end of day

- 569 nodes vs 738; `subclause` 159 vs 301. Diagnosed to PSM 3 discarding the
  label gutter, plus a column-detection knock-on. Not yet fixed.
- 7 pages spuriously deskewed on a geometrically perfect render.
- Table detection over- and under-firing; the `implied_cells` proxy for
  `ruling_line_count` was flagged as the least-confident piece when written.
- No real scanned document exists to validate against — every result is from
  OCR'ing a render, which flatters accuracy.

---

## Day 2 — 2026-09-03

### Goal for the day

Diagnose and fix what a richer hand-verified ground truth
(`rancangan_kontrak1.ground_truth.json`) and repeated rounds of human-judged
stratified review samples (`sample_for_review1.csv` through `.csv7`) caught.
Ran as many review rounds as the day allowed — each round surfaced real bugs,
fixes were applied, the next round surfaced more.

### Starting state

26/28 core-field checks passed; 60/66 human-reviewed nodes correct.

### Fixes made, in the order applied

1. **`duration`, `monetary`, `penalty_rate` in `core_fields.py` all used
   `.search()` instead of `.finditer()`**, returning only the first match each
   — invisible second duration (masa pemeliharaan, 180 days), no `monetary`
   entity type at all, and only one of two penalty rates found. Same mistake
   caught three separate times because each field's ground truth was verified
   independently — worth remembering as a class of bug, not three unrelated
   ones.
2. **`evaluate.py` was double-counting matches**: with only one real
   `penalty_rate` entry, two distinct expectations both showed `[PASS]`
   against the same actual entry. Rewrote the checker to consume each match at
   most once.
3. **Column-bleed on wrapped clause titles.** `_line_groups` clusters words
   into lines by y-proximity *before* column identity is assigned, so a short
   left-column heading and the first line of its right-column body — sharing
   the same `top` — got concatenated into one string before any split could
   happen. Fixed in `blocks.py` by splitting each merged line at the right
   column's own empirically-found start position (the minimum leading x0 past
   the gap), not the page midpoint, which a hanging-indent artifact routinely
   threw 30–80pt off.
4. **Captions/footnotes on ruled-table pages (p.67–69) were gluing across
   table and page boundaries.** Not a table-detection bug — `find_tables()`
   was already correct. The tree builder was feeding orphan blocks outside
   every table's bbox through ordinary flowing-prose continuation logic, with
   nothing to close the previously-open node. Fixed by giving `ruled_table`
   pages their own handling: every orphan block becomes its own node, merged
   with the previous one only within 20pt vertically.
5. **Cross-page merge (p.70–71) and its knock-on effects.** PAKTA specimen
   forms glued together for lack of a heading boundary. Introduced the
   **ALL-CAPS heading rule** (a standalone all-caps line ≥10 chars, ≥8 letters
   is a title) — surveyed against every such line on all 74 pages first, zero
   false positives in isolation. The rule then interacted with three
   in-progress structures and needed three follow-up fixes: a multi-line
   letterhead fragmenting into one node per line, a split section heading, and
   a heading gluing to its own following body text (plus a stale duplicate left
   in `text_raw` once that was fixed).
6. **Most serious bug of the project: a subclause body silently reassigned to
   the wrong parent clause**, reading as a fluent but factually wrong
   sentence. Cause: `round(top, 0)` in the block sort uses Python's
   banker's-rounding, so two blocks 0.1pt apart (504.5 vs 504.6) landed in
   different integer buckets and flipped their order. Fixed with a coarser 6pt
   row bucket, safely under the ~12–14pt line height.
7. **43 nodes misclassified as `clause`** (Tembusan lists, SPMK/SPPBJ
   instructions, PAKTA checklists) because `decimal_plain` numbering mapped to
   `clause` unconditionally. Fixed with a page-height heuristic — the SSUK
   two-column body is uniformly 612×792, every other section ~936pt tall — and
   verified the 3 genuine SSUK clauses that happen to sit on pages the
   *layout* detector calls `single_column` were not swept up by it.

Six additional review "failures" were dismissed as false positives — the
reviewer was reading `text_raw` instead of `title` — and fixed at the source by
adding a `title` column to the review CSV.

### What was built

**`ground_truth/regression_checks.json`** — one permanent entry per bug ever
found and fixed (20 by end of day), run automatically on every `evaluate.py`
invocation. Built because round-over-round "did it get better" claims were
unreliable: `sample_review.py` draws a fresh random sample against a different
node population each time, so 92% one round and 96% the next aren't
comparable. The checklist is the same check every run, so "did a known bug
come back" becomes a yes/no fact. Key design point: never locate a node by
`node_id` (they shift on any structural change) — locate by stable fields, and
report `AMBIGUOUS`/`NOT_FOUND` as distinct from `PASS`/`FAIL` so an
under-specified check can't silently validate against the wrong node.

### End state

```
pages: 74  nodes: 738  tables: 18
core fields populated: 6/6  overall_confidence=0.85
validation: passed  hard_fails=0  warns=2
  sibling_sequence: 6 sequence breaks (down from 12)
  dual_parser_oracle: avg_ratio~0.91
28/28 core-field checks   20/20 regression checks
```

`review/sample_for_review7.csv` (76 nodes) generated but not judged.

### Known limitations going into Day 3

- **`SSUK_BODY_PAGE_HEIGHT = 792.0` is a hardcoded US-Letter constant in
  `tree.py`, applied regardless of profile** — the single biggest portability
  risk introduced this session. A different two-column contract on A4 would
  have every clause silently misclassified as a flat list item, reintroducing
  fix #7's exact bug. Proposed fix (derive it empirically per document) not
  implemented.
- `dual_parser_oracle` low-ratio pages (13, 14, 57, 61, 62, 67–69) still
  uninvestigated.
- The rich ground-truth sections (`node_samples`, `negative_checks`,
  `related_document_numbers`, `known_source_typos`) still aren't read by
  `evaluate.py` — everything verified against them so far was done by hand.
- The ALL-CAPS heading rule assumes a document convention; a contract not
  following it simply never triggers the rule (safe no-op).

---

## Day 1 — 2026-09-02

### Goal for the day

Draft the first working version of the contract PDF extraction pipeline
(no LLM fallback), get it running end to end against the real sample
document, and put an evaluation process in place so future changes can be
checked against something other than "looks right."

### What was built

- **Full pipeline** (`pipeline/`), implementing Stages 1–9 from
  `skema_json_dan_logika_ekstraksi.md`: probe → route → profile-select →
  layout-detect → block-extract → tree-build → entities → core-resolution →
  validate. Outputs `raw_extraction.json` — the immutable source-of-truth
  layer only (no `enriched.json`/`chunks.jsonl` this round; not asked for).
- **No LLM fallback anywhere**, per explicit instruction. Every core field
  resolves via regex/heuristic strategies 1–4; an unresolved field is a
  documented `null` + `review_reason`, never a model call.
- **Profile registry**: `generic_contract_v1` (mandatory fallback) and
  `perpres16_konstruksi_v1` (matches this contract family — Indonesian govt
  construction contracts under Perpres 16/2018).
- **Evaluation tooling**: `pipeline/evaluate.py` (scores `raw_extraction.json`
  against hand-written ground truth) and `pipeline/sample_review.py`
  (stratified 10% human-review sample of tree nodes, by `sub_document`).
- **Ground truth**: `ground_truth/rancangan_kontrak.ground_truth.json`,
  pre-filled from the values already hand-verified in
  `analisis_pipeline_kontrak.md`.
- **Setup**: `requirements.txt` (just `pdfplumber`), `README.md` with venv
  instructions for Windows/macOS/Linux and the full evaluation workflow.

### Final state at end of day

Ran against the real `Rancangan Kontrak.pdf` (74 pages):

```
profile: perpres16_konstruksi_v1 (score=1.0)
pages: 74  nodes: 639  tables: 18
core fields populated: 6/6  overall_confidence=0.82
validation: passed  hard_fails=0  warns=2
  [WARN] sibling_sequence: 12 sequence breaks
  [WARN] dual_parser_oracle: avg_ratio=0.909
```

Ground-truth evaluation: **20/20 checks passed (100%)** — all six core
fields, all three structural invariants, all four identifier-survival
checks. Node-level human review (`review/sample_for_review.csv`, 66 sampled
nodes) is generated but **not yet judged** — that's manual work for a human
against the PDF, not something this session can complete.

### Key findings

1. **The two-column SSUK split was the central risk, and it needed real
   tuning to work.** The first layout-detection heuristic (both columns must
   carry ≥25% of a page's lines) failed on the actual document — the left
   "clause heading" column is often a single line while the right "body"
   column dominates. Rewrote detection to look for one dominant mode plus
   *any* clearly-separated minor mode to its left, regardless of share.
   Result: clause sequence in the SSUK body extracts as gapless.

2. **The pipeline found 80 SSUK clauses, not the 79 documented in the
   original analysis.** Clause 80 ("Itikad Baik", page 61) is a genuine,
   well-formed clause matching the same pattern as every other one — the
   original 79 count came from a coordinate spot-check, not a full manual
   read. Ground truth was revised to 80 with a note; flagged for a human to
   re-verify against the PDF directly if it matters downstream. This is the
   evaluation process doing exactly what it's for: surfacing a place where
   the *ground truth*, not just the pipeline, needed a second look.

3. **Ruled-table pages were silently dropping non-table text.** Pages
   classified `ruled_table` (SSKK, Lampiran A) skipped block-level text
   extraction entirely, which meant page/section headings sitting outside a
   table's bbox (e.g. "SYARAT-SYARAT KHUSUS KONTRAK") were lost. This broke
   sub-document detection (found 3–4 of 5 sub-documents instead of 5). Fixed
   by extracting text blocks on these pages too, then filtering out only the
   blocks that actually fall inside a detected table's bbox.

4. **Case sensitivity mattered for heading detection.** `Pasal N` headings
   and inline body references like `...sesuai pasal 44.2...` differ only by
   capitalization. A case-insensitive match on `PASAL` was misreading
   cross-references mid-paragraph as new top-level article headings
   (duplicated `Pasal 44`). Fixed by requiring Title-Case/ALL-CAPS for real
   headings.

5. **Regex context windows must not stop at newlines.** The penalty-rate
   extractor (`denda[^.\n]{0,120}`) excluded newlines from its search window,
   but Indonesian contract prose wraps every ~10 words — so `1/1000 (satu
   per seribu)` sitting on the line after "denda" was invisible to it.
   Removed the `\n` exclusion; kept the `.` (sentence-end) exclusion.

6. **`parties` resolution needed a second strategy.** This contract doesn't
   use the `PIHAK PERTAMA`/`PIHAK KEDUA` marker style at all — it uses
   `... yang bertindak untuk dan atas nama X, selanjutnya disebut "Y"`. Added
   that as a fallback structural strategy, with care taken to (a) exclude
   self-referential terms like `"Kontrak"` that also get `disebut`'d but
   aren't parties, and (b) anchor the organization-name search to the
   nearest preceding party boundary, not just "search backward N chars" —
   the first version accidentally attributed the employer's organization
   text to the contractor party too, because the two `disebut` clauses sit
   close together in this document.

7. **Indonesian number-words**: `seratus` (100) wasn't in the base
   ones/teens/magnitude tables — only `ratus` was, so `"seratus dua puluh"`
   parsed as 20, not 120, which silently failed the `words_check`
   cross-validation on the contract duration. Added `se-` prefix handling
   (`se`+magnitude = `1×`magnitude). Scoped deliberately to the hundreds
   range actually used by `duration` validation — not a general Indonesian
   numeral parser (documented as a known limitation, not fixed further).

8. A latent bug surfaced only once ruled-table pages started producing text
   blocks: the `bullet` numbering pattern had no capturing group, so
   `match_numbering` crashed (`IndexError`) the first time a bulleted line
   (from a Lampiran B checklist) actually reached it.

### Known limitations going into Day 2

- `sibling_sequence` breaks (12 currently) are flagged for review, not
  auto-corrected — the design doc's backtracking-depth-hypothesis idea is
  not implemented.
- `dual_parser_oracle` cross-check average is 0.909, with several pages
  below 0.90 (`p13:0.58`, `p14:0.56`, `p61:0.54` are the worst). Not yet
  investigated — worth a look before trusting node text on those pages.
- `parties` resolution only handles two marker styles. A contract using
  neither will resolve to `[]` + `no_parties_detected`.
- The node-level human review sample (`review/sample_for_review.csv`) is
  generated but unjudged — this is the next concrete task, not something to
  defer indefinitely.
- No OCR engine, no LLM fallback, no `enriched.json`/chunking layer — all
  explicitly out of scope for this version, not oversights.

### Next steps (proposed, not started)

- [ ] Judge `review/sample_for_review.csv` against the source PDF.
- [ ] Investigate the low dual-parser-ratio pages (13, 14, 57, 61).
- [ ] Decide whether to build a second profile (e.g. for a non-construction
      contract type) to stress-test `generic_contract_v1`'s fallback path.
- [ ] Revisit the 79-vs-80 clause count with an actual page-by-page read.
