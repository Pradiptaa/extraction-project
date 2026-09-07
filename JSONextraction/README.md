# Contract PDF Extraction — v1

Extracts a born-digital Indonesian government contract PDF into
`raw_extraction.json`: the schema-agnostic core fields + generic recursive
node tree + ruled tables + entities described in
[`skema_json_dan_logika_ekstraksi.md`](skema_json_dan_logika_ekstraksi.md),
built per the technical plan in
[`analisis_pipeline_kontrak.md`](analisis_pipeline_kontrak.md).

## Scope of this version

This is the extraction layer only (Stages 1–9: probe → route → profile
select → layout detect → block extract → tree build → entities → core
resolution → validate). It does **not** include:

- **No LLM fallback.** Strategy 5 from the entity cascade is not implemented.
  A core field that clears no regex/heuristic strategy's confidence
  threshold resolves to `value: null` with a `review_reason` — a valid,
  documented outcome, not a crash.
- **No OCR engine wired in.** Pages that would need OCR (near-zero text +
  high image coverage) are flagged `ocr_needed` in `pages[].extraction_method`
  and left with empty text, rather than silently dropped or guessed at. None
  of the 74 pages in `Rancangan Kontrak.pdf` need this — it's a complete
  born-digital text layer.
- **No preprocessing/derivation layer or chunker.** `enriched.json`
  (stemmed/embedding text views) and `chunks.jsonl` (retrieval units) are not
  built by this version — only the immutable source-of-truth
  `raw_extraction.json`.
- **Sequence-break flagging, not backtracking.** When a sibling numbering
  sequence breaks (e.g. `37`, `38`, `40`), the depth-inference design doc
  proposes backtracking to an alternate depth hypothesis. This version flags
  the break (`sibling_sequence` warning) for human review instead.

## What it does

- Native text extraction via `pdfplumber`, with word-level coordinates.
- Per-page layout classification (`single_column` / `two_column` /
  `ruled_table` / `form` / `mixed` / `blank`) from a word-x0 histogram —
  every threshold is a *fraction* of page width, so it isn't broken by mixed
  page sizes in one document.
- Coordinate-based two-column splitting (row-major reading order), which is
  what fixes the reading-order collapse that naive `PdfReader.extract_text()`
  produces on a borderless two-column contract body.
- Ruled-table extraction via `pdfplumber`'s cell reconstruction for pages
  with ruling lines (e.g. SSKK), instead of flat text that bleeds columns.
- A generic, label-agnostic numbering/tree builder (`part`, `article`,
  `section`, `clause`, `subclause`, `list_item`, ...), with page-break
  stitching.
- A profile registry (`profiles/*.json`): `generic_contract_v1` (mandatory
  fallback) and `perpres16_konstruksi_v1` (Indonesian govt construction
  contracts, matches the sample PDF). Profiles supply sub-document markers
  and validation invariants only — no code changes needed for a new
  contract family.
- An entity cascade (regex + gazetteer; Strategies 1–4) that promotes
  candidates into the six guaranteed `core` fields: `document_type`,
  `contract_name`, `contract_number`, `parties`, `key_dates`, `key_numbers`.
- Indonesian-aware normalizers: currency (`.`=thousands, `,`=decimal), dates
  (Indonesian month names), number-words (`Seratus Dua Puluh` → `120`, for
  cross-validating written numbers against digits), and rate unification
  (`‰`, `%`, `x/y`).
- A validation gate (`quality` block in the output): core presence/typing,
  character conservation, tree integrity, duplicate-span detection,
  identifier survival, encoding sanity, placeholder tagging, words-vs-digits
  agreement, and profile-declared invariants (e.g. gapless clause sequence).
  An optional dual-parser cross-check runs automatically if `pdftotext`
  (Poppler) is on `PATH`.

## Setup

Requires Python 3.10+ (uses `X | None` union syntax and `dataclass` slots
throughout).

### Windows (PowerShell)

```powershell
cd JSONextraction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux (bash)

```bash
cd JSONextraction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# with the venv activated
python -m pipeline.main "Rancangan Kontrak.pdf" --out output
```

This writes `output/raw_extraction.json` and prints a summary:

```
profile: perpres16_konstruksi_v1 (score=1.0)
pages: 74  nodes: 625  tables: 18
core fields populated: 6/6  overall_confidence=0.808
validation: passed  hard_fails=0  warns=2
  [WARN] sibling_sequence: 12 sequence breaks
  [WARN] dual_parser_oracle: avg_ratio=0.893 low_pages=[...]
wrote output\raw_extraction.json
```

The process exit code is `0` if validation passed, `2` if any hard-fail
check failed (the JSON is still written either way — a hard fail is a signal
to gate downstream ingestion, not a crash).

Optional: install Poppler's `pdftotext` and put it on `PATH` to enable the
dual-parser cross-check (`brew install poppler` / `apt install poppler-utils`
/ the Windows Poppler binaries). Without it, that check is skipped, not
failed.

To run against a different contract, point at a different PDF:

```bash
python -m pipeline.main "path/to/other_contract.pdf" --out output
```

If it doesn't match `perpres16_konstruksi_v1`'s signals closely enough, it
falls back to `generic_contract_v1` automatically — core fields still get
populated, just with a shallower structure tree.

## Evaluation & ground truth

The `quality` block already embedded in `raw_extraction.json` only proves
**self-consistency** — the tree doesn't contradict itself, IDs resolve,
characters aren't dropped. It cannot tell you whether the *content* is
actually right, because it has nothing to compare against. That needs a
separate ground-truth check, which is what `pipeline/evaluate.py` and
`pipeline/sample_review.py` are for. Two independent checks, because they
answer different questions:

### 1. Core-field & structural accuracy — `ground_truth/*.ground_truth.json`

A small hand-written JSON file records the *known-correct* values for one
document: `contract_number`, the employer's representative name and NIP, key
dates, the duration, and document-level facts like the clause count. Run:

```bash
python -m pipeline.evaluate output/raw_extraction.json \
    --ground-truth ground_truth/rancangan_kontrak.ground_truth.json
```

It prints a PASS/FAIL line per field, e.g.:

```
[PASS] core.contract_number: expected='08/PUPRPRKP-B.PNK/SP-PPK' actual='08/PUPRPRKP-B.PNK/SP-PPK'
[FAIL] structural.clause_count_general_terms: expected=79±0 actual=80
```

**Building ground truth for a new document**: copy
`ground_truth/rancangan_kontrak.ground_truth.json`, read the source PDF
yourself, and fill in the `expected_value`/`expected_amount`/`expected_date`
fields under `core` with what the PDF actually says — not what the pipeline
extracted. The whole point is an independent reference; copying the
pipeline's own output back into the ground-truth file makes the check
circular and worthless. `identifier_survival_checks` is the cheapest part to
fill in: just list a handful of strings (contract number, NIP, any reference
number) that must appear byte-for-byte in the extracted text.

For this sample document, `ground_truth/rancangan_kontrak.ground_truth.json`
was pre-filled from the values already hand-verified in
`analisis_pipeline_kontrak.md` (sections A.1, A.7, F.2) — but one entry
(`clause_count_general_terms`) was revised from 79 to 80 *during* building
this eval script, because the pipeline consistently found a genuine,
well-formed 80th clause the original manual spot-check had missed. That's
the eval script doing its job: it doesn't just grade the pipeline, it also
surfaces places where your ground truth itself needs a second look — treat a
FAIL as "go look at the PDF," not "the pipeline is wrong" or "the ground
truth is wrong," until you've actually looked.

### 2. Node-level text accuracy — stratified human review sample

The core-field check only covers six fields; it says nothing about whether
the other 600+ tree nodes' `text_raw` actually matches the PDF. For that,
`sample_review.py` implements the stratified-sampling half of the design
doc's I.6 review process:

```bash
python -m pipeline.sample_review output/raw_extraction.json \
    --out review/sample_for_review.csv --fraction 0.10 --seed 42
```

This picks ~10% of nodes, stratified by `sub_document` (so a sparse section
like `annex_a` still gets a few rows instead of being drowned out by the
569-node `general_terms` section), and writes a CSV with columns `node_id`,
`sub_document`, `node_type`, `label`, `pages`, `text_raw`, and three blank
columns: `correct_yn`, `corrected_text`, `notes`.

**Filling it in**: open the CSV next to the source PDF. For each row, jump to
the listed page(s), find the corresponding clause/section, and compare its
printed text against `text_raw`. Write `Y` or `N` in `correct_yn`; if `N`,
paste the correct text into `corrected_text` and note what went wrong (wrong
column split, merged rows, truncated continuation, etc.) in `notes`. Nothing
in this repo can do this step for you — it's a manual read against the PDF,
by design; a tool "verifying itself" against its own output isn't a review.

Once judged, score it:

```bash
python -m pipeline.evaluate output/raw_extraction.json \
    --ground-truth ground_truth/rancangan_kontrak.ground_truth.json \
    --review-csv review/sample_for_review.csv
```

This reports completeness (how many rows are still blank), node accuracy
(`% correct` among judged rows, target ≥99% per the design doc's I.6), and
lists every row marked `N` with its page/label/notes so you can go straight
to the bug instead of re-scanning the whole CSV. `evaluate.py` exits `0`
only when every core-field check passes **and** the review CSV, if given, is
fully judged with zero `N`s — a review sample sitting half-filled-in reports
`FAIL`, deliberately, so a stale or incomplete review can't be mistaken for
a passing one.

### Re-running after a pipeline change

Any time you touch `pipeline/*.py`, re-run `pipeline.main`, then
`pipeline.evaluate` against the same ground-truth file. A regression shows
up as a check flipping from PASS to FAIL — that's the whole reason this
exists as a script instead of a one-off manual comparison.

### The bug regression checklist

`sample_review.py` draws a *fresh random sample* every time it runs, against
whatever the tree looks like right now — the node population changes as the
pipeline changes, so a 90% this round and a 96% last round aren't actually
comparable numbers, and a bug that was fixed three rounds ago has no
guarantee of being resampled to confirm it's still fixed. That's what
`ground_truth/regression_checks.json` is for: one permanent, accumulating
entry per bug ever found and fixed, checked the *same* way every single run.
`pipeline.evaluate` runs it automatically (default path
`ground_truth/regression_checks.json`; pass `--regression-checks` to point
elsewhere, or to a nonexistent path to skip it).

Each entry locates a specific node by stable fields (`sub_document` +
`node_type` + `label_normalized`, or `pages_contains` + a distinctive
`text_raw_contains`/`title_contains`/`text_raw_equals` substring for nodes
without a label) and asserts something about it (`title_equals`,
`text_raw_equals`, `has_child`, or a `kind: "count"` entry for "how many
nodes match this pattern"). Never `node_id` — those are assigned
sequentially and shift on any structural change, so pinning one would make
the check pass or fail based on an accident of ordering rather than content.

A locate block that doesn't resolve to exactly one node (for `kind: "node"`
entries) reports `NOT_FOUND` or `AMBIGUOUS` instead of PASS/FAIL — both still
count as failures, but distinctly, because a check examining the wrong node
(or no node) is worse than useless: it can silently validate against
whatever happened to match first. This isn't theoretical — building this
file, `label_normalized: "b"` alone matched two unrelated section headings
in different sub-documents (one in the SSUK body, one in a table of contents
inside Lampiran B), and a `text_raw_contains: "Pekerjaan Utama"` locate
matched both "Pekerjaan Utama" and "Pekerjaan **bukan** Pekerjaan Utama."
Both were caught as AMBIGUOUS immediately rather than silently passing
against the wrong node.

**Adding a new entry**: when you find and fix a new bug, add one entry to
`regression_checks.json` before moving on — that's what keeps the list
monotonic instead of static. Locate the affected node as specifically as
you can, and if you're not sure whether a locate is unique, that uncertainty
is exactly what running it will tell you (AMBIGUOUS means it isn't).

## Layout

```
JSONextraction/
  pipeline/
    probe.py        Stage 1 — per-page geometry/font/image signals
    router.py        Stage 2 — native vs. OCR-needed routing (no OCR engine)
    profiles.py       Stage 3 — profile scoring/selection
    layout.py         Stage 4 — per-page layout classification
    blocks.py         Stage 5 — ordered text blocks + ruled-table extraction
    numbering.py       numbering-token recognizer (style-agnostic)
    tree.py           Stage 6 — recursive node tree build
    entities.py        Stage 7 — regex/gazetteer entity cascade
    core_fields.py      Stage 8 — core field resolution (Strategies 1-4)
    validate.py        Stage 9 — validation gate
    normalize.py       Indonesian currency/date/number-word/rate parsing
    schema.py          value-object + ID helpers
    main.py           CLI orchestrator
    evaluate.py        scores raw_extraction.json against ground truth
    sample_review.py    builds the stratified node-review CSV
  profiles/
    generic_contract_v1.json
    perpres16_konstruksi_v1.json
  ground_truth/
    rancangan_kontrak.ground_truth.json   hand-verified expected values
    regression_checks.json                permanent per-bug checklist (grows over time)
  review/
    sample_for_review.csv                 human-review sample (fill in correct_yn)
  requirements.txt
  output/             raw_extraction.json lands here (gitignored)
```

## Known limitations / next steps

- `sibling_sequence` breaks are flagged, not auto-corrected — review the
  `quality.tree_quality_flags` list and the flagged nodes' `bbox` against the
  source PDF.
- The party-extraction structural strategy (`core_fields.resolve_parties`)
  handles the `PIHAK PERTAMA`/`PIHAK KEDUA` marker style and the `... atas
  nama X, selanjutnya disebut "Y"` style. A contract using neither pattern
  will resolve `parties` to `[]` with `no_parties_detected` — a genuine gap
  to close with another profile-specific strategy, not silently guessed at.
- `entities.py`'s number-word parser (`parse_number_words_id`) is correct for
  the hundreds range used by `duration` cross-validation; it is not a
  general Indonesian numeral parser (compound magnitudes like "lima ratus
  juta" are out of scope — nothing in the core-field cascade currently needs
  million-scale word parsing).
