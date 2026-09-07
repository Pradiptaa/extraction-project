# Contract PDF Extraction

Extracts an Indonesian government contract PDF into `raw_extraction.json`: the
schema-agnostic core fields + generic recursive node tree + ruled tables +
entities described in
[`skema_json_dan_logika_ekstraksi.md`](skema_json_dan_logika_ekstraksi.md),
built per the technical plan in
[`analisis_pipeline_kontrak.md`](analisis_pipeline_kontrak.md). A second stage
reduces that into `clean_extraction.json` — a small, keyword-only summary
meant for storage and search at scale. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for a file-by-file map of how the pieces
fit together, and [`PROGRESS_LOG.md`](PROGRESS_LOG.md) for how each part came
to be built the way it is.

## Two extraction pipelines, one JSON schema

| | Native (`pipeline.main`) | OCR (`pipeline.ocr_main`) |
|---|---|---|
| Input | Born-digital PDF with a real text layer | Scanned / image-only PDF |
| Text source | `pdfplumber` word extraction | Tesseract, rendered via PyMuPDF |
| Tables | `pdfplumber` vector-line cell reconstruction | OpenCV rule-grid detection |
| Speed | Seconds | Minutes (two OCR passes per page) |

Both produce the exact same `raw_extraction.json` shape. This isn't a
coincidence: the shared stages (layout → blocks → tree → entities →
core_fields → validate) only ever consume a `PageProbe` — a page's words, each
with a bounding box — and never touch a PDF directly. The OCR pipeline
reimplements just the two PDF-specific pieces (`probe_document` and table
extraction) and **imports** everything downstream, unmodified. A fix to
`tree.py` benefits both pipelines automatically; neither can silently drift
from the other's output shape.

The OCR path has only ever been validated against a render of the same
born-digital sample PDF — no genuinely scanned document has been available to
test against. Structural correctness (node counts, table detection, column
splitting) has been verified to match native output; character-level OCR
accuracy on a real scan is untested and will be lower than what's reported
here.

## Scope

- **No LLM fallback, anywhere.** Every core field resolves via regex/heuristic
  strategies. An unresolved field is a documented `value: null` with a
  `review_reason` — a valid, expected outcome, never a model call.
- **No preprocessing/derivation layer beyond `clean_extraction.json`.** No
  chunker, no embedding-text view.
- **Sequence-break flagging, not backtracking.** A broken sibling numbering
  sequence (e.g. `37`, `38`, `40`) is flagged (`sibling_sequence` warning) for
  human review, not auto-corrected via an alternate depth hypothesis.

## What the extraction stage does

- Per-page layout classification (`single_column` / `two_column` /
  `ruled_table` / `form` / `mixed` / `blank`) from a word-x0 histogram — every
  threshold is a *fraction* of page width/height, so mixed page sizes within
  one document don't break it.
- Coordinate-based two-column splitting in true row-major reading order, which
  is what a borderless two-column contract body needs and naive text
  extraction can't give you.
- Ruled-table extraction for pages with ruling lines (native: `pdfplumber`
  cell reconstruction; OCR: OpenCV morphological rule detection, qualified
  into an actual grid rather than counted — a bordered box or a letterhead
  emblem doesn't count as a table).
- A generic, label-agnostic numbering/tree builder (`part`, `article`,
  `section`, `clause`, `subclause`, `list_item`, ...), with page-break
  stitching.
- A profile registry (`profiles/*.json`): `generic_contract_v1` (mandatory
  fallback) and `perpres16_konstruksi_v1` (Indonesian govt construction
  contracts, matches the sample PDF). Profiles supply sub-document markers and
  validation invariants only — no code changes needed for a new contract
  family.
- An entity cascade (regex + gazetteer) promoting candidates into the six
  guaranteed `core` fields: `document_type`, `contract_name`,
  `contract_number`, `parties`, `key_dates`, `key_numbers`.
- Indonesian-aware normalizers: currency (`.`=thousands, `,`=decimal), dates
  (Indonesian month names), number-words (`Seratus Dua Puluh` → `120`), and
  rate unification (`‰`, `%`, `x/y`).
- A validation gate (`quality` block in the output): core presence/typing,
  character conservation, tree integrity, duplicate-span detection, identifier
  survival, encoding sanity, placeholder tagging, words-vs-digits agreement,
  and profile-declared invariants. An optional dual-parser cross-check against
  Poppler's `pdftotext` runs automatically on the native path if it's on
  `PATH` (not applicable on the OCR path — there's no independent native text
  layer to compare against).

## What the keyword-extraction stage does

`keywords/clean_json.py` reduces `raw_extraction.json` to `clean_extraction.json`
— roughly 0.7% the size, meant for downstream storage/search where the full
node tree is unnecessary overhead. It never imports from `pipeline/`, so it
runs identically on native and OCR output.

- Core fields are flattened from their `{value, raw, confidence, ...}`
  wrapper to plain values.
- A `body` field of search keywords: the resolved core fields (contract name,
  number, parties, reference numbers) are seeded in as guaranteed terms, then
  filled out with statistically mined phrases from the node tree's own text.
- Two interchangeable mining backends, `keywords/extractor.py`:
  - **YAKE** (default) — unsupervised, single-document, no model or corpus
    required.
  - **RAKE** — implemented from scratch (not `rake-nltk`, which assumes an
    English-centric tokenizer/corpus this project doesn't need) using the same
    Indonesian tokenizer and stopword list as YAKE.
  - Both are single-document, frequency-based algorithms — neither can tell
    "frequent because it's boilerplate contract-template language" apart from
    "frequent because it matters to this case." That distinction needs a
    multi-document corpus to compare against, which isn't available yet with
    one sample contract. Clause/article/section **titles** are excluded from
    mining for this reason — in this profile family they're fragments of a
    fixed external outline (the SSUK/SSKK table of contents), not authored
    content; their body text (`text_raw`) is still mined.
- An Indonesian stopword list (`keywords/stopwords_id.txt`, 757 terms from
  [stopwords-iso](https://github.com/stopwords-iso/stopwords-id)), adjusted
  for this domain: `pihak`/`waktu`/`bagian` are rescued (load-bearing in a
  contract, wrongly generic in a prose-tuned list), contract boilerplate and
  structural vocabulary (`pasal`, `ayat`, `lampiran`, ...) are added.

## Setup

Requires Python 3.10+ (uses `X | None` union syntax throughout).

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

`pip install` covers the native pipeline and keyword extraction fully. The OCR
pipeline additionally needs the **Tesseract binary itself** plus the `ind` and
`eng` traineddata files — pip cannot install these:

- Windows: the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
  — tick "Additional language data" → Indonesian during setup, or drop
  `ind.traineddata` into `Tesseract-OCR\tessdata\` afterward.
- macOS: `brew install tesseract tesseract-lang`
- Linux: `apt install tesseract-ocr tesseract-ocr-ind`

Verify with `tesseract --list-langs` (must show both `ind` and `eng`). If
Tesseract isn't on `PATH`, pass `--tesseract-cmd` to `pipeline.ocr_main` (see
below).

Optional, native path only: install Poppler's `pdftotext` and put it on `PATH`
to enable the dual-parser cross-check (`brew install poppler` /
`apt install poppler-utils` / the Windows Poppler binaries). Without it, that
check is skipped, not failed.

## Run

### Native pipeline (born-digital PDF)

```bash
python -m pipeline.main "Rancangan Kontrak.pdf" --out output
```

```
profile: perpres16_konstruksi_v1 (score=1.0)
pages: 74  nodes: 738  tables: 18
core fields populated: 6/6  overall_confidence=0.85
validation: passed  hard_fails=0  warns=2
  [WARN] sibling_sequence: 6 sequence breaks
  [WARN] dual_parser_oracle: avg_ratio=0.954 low_pages=[...]
wrote output\raw_extraction.json
```

### OCR pipeline (scanned PDF)

```bash
python -m pipeline.ocr_main "scan.pdf" --out output_ocr
```

Expect roughly 1–3 seconds per page (two Tesseract passes run per page — see
`ARCHITECTURE.md` for why). Useful flags while tuning a run:

| Flag | Effect |
|---|---|
| `--dpi 400` | Higher render resolution (default 300), usually better on poor scans |
| `--single-pass` | Skip the recovery pass — ~2x faster, but drops numbering labels the first pass misses |
| `--no-deskew` | Skip tilt correction |
| `--debug-dir debug_ocr` | Dump each preprocessed page as PNG — first thing to check if a page comes back empty |
| `--min-conf 0` | Keep every word Tesseract emits, to check whether the confidence floor is discarding real text |
| `--tesseract-cmd PATH` | Point at `tesseract.exe` if it isn't on `PATH` |

Both pipelines share `--profile-dir` (override the profile registry location).
The exit code is `0` if validation passed, `2` if any hard-fail check failed —
the JSON is written either way; a hard fail is a signal to gate downstream
ingestion, not a crash.

To run against a different contract, point at a different PDF. If it doesn't
match `perpres16_konstruksi_v1`'s signals closely enough, it falls back to
`generic_contract_v1` automatically — core fields still populate, just with a
shallower structure tree.

### Keyword extraction (either pipeline's output)

```bash
python -m keywords.clean_json output\raw_extraction.json --out output
```

```
document: Peningkatan Jalan Mekar Desa Natai Sedawak
method:   native extraction, yake keywords, profile=perpres16_konstruksi_v1
body:     44 keywords
size:     1,109,186 -> 8,169 bytes (0.7% of raw)
wrote output\clean_extraction.json
```

`--method rake` swaps the mining backend (see above); `--top-n 60` raises the
cap on mined keywords (default 40, seeded core-field terms don't count against
it). Works identically on `output_ocr\raw_extraction.json`.

## Evaluation & ground truth

The `quality` block already embedded in `raw_extraction.json` only proves
**self-consistency** — the tree doesn't contradict itself, IDs resolve,
characters aren't dropped. It cannot tell you whether the *content* is
actually right, because it has nothing to compare against. That needs a
separate ground-truth check: `pipeline/evaluate.py`, which runs three
independent things in one invocation.

```bash
python -m pipeline.evaluate output\raw_extraction.json \
    --ground-truth ground_truth\rancangan_kontrak1.ground_truth.json
```

```
[PASS] core.contract_number: expected='08/PUPRPRKP-B.PNK/SP-PPK' actual='08/PUPRPRKP-B.PNK/SP-PPK'
[PASS] structural.clause_count_general_terms: expected=80±0 actual=80
28/28 checks passed (100.0%)
[PASS] regression[bug_022_clause72_subclause_not_misparented]: OK
20/20 regression checks passed
RESULT: PASS
```

`rancangan_kontrak1.ground_truth.json` is the actively-used, hand-verified
ground truth (per-field `_verification` notes citing page numbers and quotes).
`rancangan_kontrak.ground_truth.json` is the original, simpler v1 file, kept
for reference.

### 1. Core-field & structural accuracy

Hand-written JSON records the *known-correct* values for one document:
`contract_number`, the employer's representative name and NIP, key dates, the
duration, document-level facts like clause count, and identifier-survival
checks (byte-for-byte substrings that must appear in the extracted text).

**Building ground truth for a new document**: copy the ground-truth file, read
the source PDF yourself, and fill in the expected values with what the PDF
actually says — not what the pipeline extracted. Copying the pipeline's own
output back into the ground-truth file makes the check circular and
worthless.

### 2. The bug regression checklist

`ground_truth/regression_checks.json` — one permanent, accumulating entry per
bug ever found and fixed (20 currently), checked the *same* way every run.
This exists because `sample_review.py` draws a fresh random sample each time
against whatever the tree looks like right now, so a 90% this round and a 96%
last round aren't comparable numbers, and a bug fixed three rounds ago has no
guarantee of being resampled to confirm it's still fixed. Runs automatically
(default path `ground_truth/regression_checks.json`; `--regression-checks` to
point elsewhere or to skip it).

Each entry locates a node by stable fields (`sub_document` + `node_type` +
`label_normalized`, or `pages_contains` + a distinctive text substring) —
never `node_id`, which is assigned sequentially and shifts on any structural
change. A locate resolving to anything other than exactly one node reports
`AMBIGUOUS`/`NOT_FOUND` rather than `PASS`/`FAIL`, since a check silently
validating against the wrong node is worse than useless.

**Adding a new entry**: when you find and fix a new bug, add one entry before
moving on — that's what keeps the list monotonic. If a locate's uniqueness is
in doubt, running it will tell you (`AMBIGUOUS` means it isn't unique yet).

### 3. Node-level text accuracy — stratified human review sample

The core-field check only covers six fields; it says nothing about whether
the other 700+ tree nodes' `text_raw` actually matches the PDF.

```bash
python -m pipeline.sample_review output\raw_extraction.json \
    --out review\sample_for_review.csv --fraction 0.10 --seed 42
```

Picks ~10% of nodes, stratified by `sub_document` (so a sparse section still
gets a few rows instead of being drowned out by the largest one), and writes
a CSV with blank `correct_yn`/`corrected_text`/`notes` columns. Open it next
to the source PDF, judge each row by hand — nothing in this repo can do that
step for you, a tool "verifying itself" against its own output isn't a
review — then score it:

```bash
python -m pipeline.evaluate output\raw_extraction.json \
    --ground-truth ground_truth\rancangan_kontrak1.ground_truth.json \
    --review-csv review\sample_for_review7.csv
```

This reports completeness and node accuracy (target ≥99%), and lists every
row marked `N` with its page/label/notes. `evaluate.py` exits `0` only when
every core-field check passes **and** the review CSV, if given, is fully
judged with zero `N`s — a half-filled review reports `FAIL` deliberately, so
a stale review can't be mistaken for a passing one.

### Re-running after a change

Any time you touch `pipeline/*.py` or `keywords/*.py`, re-run the relevant
pipeline, then `pipeline.evaluate` against the same ground-truth file. A
regression shows up as a check flipping from PASS to FAIL.

## Layout

```
JSONextraction/
  pipeline/
    probe.py         Stage 1 — per-page geometry/font/image signals (native)
    router.py         Stage 2 — native vs. OCR-needed routing
    profiles.py        Stage 3 — profile scoring/selection
    layout.py          Stage 4 — per-page layout classification (shared)
    blocks.py          Stage 5 — ordered text blocks + ruled-table extraction (shared)
    numbering.py        numbering-token recognizer (shared)
    tree.py            Stage 6 — recursive node tree build (shared)
    entities.py         Stage 7 — regex/gazetteer entity cascade (shared)
    core_fields.py       Stage 8 — core field resolution (shared)
    validate.py         Stage 9 — validation gate (shared)
    normalize.py        Indonesian currency/date/number-word/rate parsing
    schema.py           value-object + ID helpers
    main.py            CLI orchestrator — native pipeline
    ocr_main.py          CLI orchestrator — OCR pipeline (render/deskew/OCR,
                        then imports the "shared" stages above unmodified)
    evaluate.py          scores raw_extraction.json against ground truth
    sample_review.py      builds the stratified node-review CSV
  keywords/
    stopwords_id.txt     757-term Indonesian stopword list
    extractor.py         YAKE/RAKE mining, seeding, stopword handling
    clean_json.py         builds clean_extraction.json; CLI
  profiles/
    generic_contract_v1.json
    perpres16_konstruksi_v1.json
  ground_truth/
    rancangan_kontrak1.ground_truth.json   active, rich hand-verified ground truth
    rancangan_kontrak.ground_truth.json    original v1, kept for reference
    regression_checks.json                 permanent per-bug checklist
  review/
    sample_for_review7.csv                 latest human-review sample
  requirements.txt
  output/             native pipeline output lands here (gitignored)
  output_ocr/         OCR pipeline output lands here (gitignored)
```

## Known limitations / next steps

- `sibling_sequence` breaks are flagged, not auto-corrected — review the
  `quality.tree_quality_flags` list and the flagged nodes' `bbox` against the
  source PDF.
- The party-extraction structural strategy (`core_fields.resolve_parties`)
  handles the `PIHAK PERTAMA`/`PIHAK KEDUA` marker style and the `... atas
  nama X, selanjutnya disebut "Y"` style. A contract using neither pattern
  resolves `parties` to `[]` with `no_parties_detected`.
- `entities.py`'s number-word parser is correct for the hundreds range used by
  `duration` cross-validation; it is not a general Indonesian numeral parser.
- **OCR path, untested against a real scan**: `contract_value` can be
  fabricated from OCR misreads of unfilled template placeholders (e.g.
  `NILAI PERJANJIAN .........` → `00`), and the government letterhead crest
  can OCR into low-confidence garbage that fragments the letterhead into extra
  nodes. Neither is fixed — the obvious fix (a confidence floor) would also
  discard legitimate short numbering labels, so it needs a geometric approach
  (suppress tokens inside detected image regions) instead. See
  `ARCHITECTURE.md` and `PROGRESS_LOG.md` for the full diagnosis.
- **Keyword extraction, both backends**: YAKE and RAKE are single-document and
  frequency-based, so `body` still leans toward contract-template language
  that recurs across any document using this profile, not just what's unique
  to this case. Fixing that properly needs a multi-document corpus to measure
  rarity against (TF-IDF), which isn't available yet with one sample
  contract.
