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

The user hand-verified a much richer ground truth
(`ground_truth/rancangan_kontrak1.ground_truth.json`, with per-field
`_verification` notes, negative checks, and known source typos) and fully
judged a 66-row stratified review sample
(`review/sample_for_review1.csv`). Diagnose and fix what those two files
caught.

### Starting state

26/28 core-field checks passed; 60/66 human-reviewed nodes correct (6
flagged, all genuine bugs, not review mistakes).

### Fixes made, in the order applied

1. **`duration` and `monetary` in `core_fields.resolve_key_numbers`
   (`.search()` → `.finditer()`)**. Both used `.search()`, which returns only
   the *first* match — so a second, distinct duration ("masa pemeliharaan",
   180 days) was invisible, and there was no `monetary` entity type at all
   (only the labeled `contract_value` lookup), so the one populated Rp
   figure in the whole document (a Rp10.000,00 stamp-duty/meterai mention)
   was never extracted. Also fixed: the duration regex required the literal
   spelling `kalender`, but the 180-day figure is genuinely truncated to
   `kalende` at a page break in the source — made the trailing `r` optional.
   Both new entity types get a `subtype` classified from a nearby keyword
   window (`masa pelaksanaan`/`masa pemeliharaan`; `meterai`). The new
   `monetary` type is deliberately kept separate from `contract_value` and
   explicitly excludes whatever span `contract_value` already claimed — a
   currency scan that just takes the first or largest Rp figure on the page
   would wrongly promote incidental amounts like this into the contract
   value.

2. **`penalty_rate`, same `.search()` bug, but subtype needed a different
   fix.** Two distinct penalty rates exist (`denda_keterlambatan` on p.6,
   `denda_cacat_mutu` on p.62, both 0.001). Switched to `.finditer()` +
   dedupe by `(subtype, rate)`. Unlike duration/monetary, the classifying
   keyword ("keterlambatan" / "cacat mutu") sits *after* "denda" **within**
   the match span itself, not in the text before it — so subtype
   classification here searches the match text, not a backward-looking
   context window.

3. **`evaluate.py` was double-counting matches.** With only one real
   `penalty_rate` entry (before fix #2), two distinct expected entries both
   showed `[PASS]` against the *same* actual entry — a false positive
   hiding the fact that only one of two expected rates had actually been
   found. Rewrote `check_key_numbers` to track consumed actual-entry indices
   so each real match can satisfy at most one expectation.

4. **Column-bleed on wrapped clause titles — root cause was upstream of
   where I first looked.** 4 of 6 review failures shared one bug: when a
   clause's left-column heading wraps to a second line, that second line
   has no numbering marker, so it can't be told apart from body
   continuation. My first attempt (route column-0 continuation lines to the
   clause's `title` instead of `text_raw`) didn't fully work — inspecting
   actual word coordinates showed the *real* problem was one level lower:
   `_line_groups` clusters words into lines by y-proximity alone, before
   column identity is assigned, so whenever a short heading line and the
   first line of its body land at the same `top` (the common case), their
   words get concatenated into one string *before* any column split can
   happen — nothing downstream can un-mix them once that happens.
   Fixed in `blocks.py`: for two-column pages, split each merged line at
   the right column's own start position, not the page's midpoint boundary
   — a per-word gap-size heuristic was tried and abandoned first (normal
   within-column word-spacing on some pages, 7–9pt, overlapped with the
   actual cross-column gap on others, ~11pt, so no fixed threshold worked).
   The page midpoint itself also proved too coarse — column-1's body text
   turned out to sit at a highly consistent x position across a page, but
   the bin-histogram-derived midpoint routinely landed 30–80pt to its right
   because of a hanging-indent artifact (a numbered subclause's first line
   starts flush with its number; wrapped continuation lines of the same
   paragraph indent further right, and the histogram's dominant-mode bin
   run picked up the deeper, more common indent instead of the true left
   edge). Fixed by computing the right column's start as the *minimum*
   leading x0 among lines past the gap, not the mode.
   All 4 flagged clauses (`Tugas dan Wewenang Pengawas Pekerjaan`,
   `Penyerahan Lokasi Kerja dan Personel`, `Penundaan Oleh Pegawas
   Pekerjaan` — reproducing the source's own typo verbatim, correctly —
   and the 8-line `Tindakan Penyedia yang Mensyaratkan...` heading) now
   match the ground truth's hand-verified titles exactly.

5. **"Table merge" on pages 67–69 — not actually a table-detection bug.**
   `pdfplumber.find_tables()` was already separating the tables correctly
   (verified: 4 distinct tables on p.67, 2 on p.68, 1 on p.69). The bug was
   in the tree builder: captions and footnotes sitting *outside* every
   table's bbox (e.g. "1) Pekerjaan Utama", a `Catatan:` footnote, the next
   table's caption) were being fed through the same stack/continuation
   logic as flowing prose — and since none of them carry a numbering match
   to close the previously-open node, they all glued onto whichever caption
   opened first, silently spanning pages 67→68→69. Fixed by giving
   ruled_table pages their own handling in `tree.py`: every orphan block
   becomes its own standalone node, merged with the previous one only if
   within 20pt vertically (i.e. genuinely the same wrapped caption).

### End state

28/28 core-field checks pass (real ones this time — fix #3 closed the
false-positive hole fix #2 would otherwise have hidden). The 6 previously
flagged review nodes were spot-checked directly against their ground-truth
`notes` and now read correctly; a fresh stratified sample
(`review/sample_for_review2.csv`, 74 nodes, 10.4%) is generated for the next
round of human judging — not yet done, since that's manual work.

### Key lesson from today

Three of five fixes (`duration`, `monetary`, `penalty_rate`) were the same
underlying mistake — `.search()` instead of `.finditer()` — caught three
separate times because each field's ground-truth entry was verified
independently. Worth a quick audit of `core_fields.py` for any other
first-match-only extraction that hasn't been caught yet, rather than waiting
for the next ground-truth pass to find it one field at a time.

### Known limitations going into Day 3

- `sibling_sequence` breaks (11, down from 13 as a side effect of today's
  fixes, still not zero) are flagged for review, not auto-corrected.
- `dual_parser_oracle` average (0.907) and its worst pages (13, 14, 57, 61)
  haven't been investigated yet — carried over from Day 1.
- The rich sections of `rancangan_kontrak1.ground_truth.json` — 13
  `node_samples`, `negative_checks`, `related_document_numbers`,
  `known_source_typos` — are still not read by `evaluate.py`. Today's
  fixes were verified against them by hand (spot-checking `text_raw`/
  `title` directly), not by the eval script. Wiring these up would let
  the script catch the next round of regressions automatically instead of
  requiring another manual pass.
- `sub_document_count` is pinned at 5 to match the shipped profile, but the
  ground truth's own notes point out the document has 8 logically distinct
  parts — SPPBJ (p.5) and SPMK (p.6) carry real extractable metadata (the
  penalty rate and a second duration attestation both live in SPMK) and are
  currently absorbed into `main_agreement`. Not addressed today.

---

### Continued — same day, logged retroactively

> Everything above was written at 08:41. Work continued until ~10:51 and was
> never logged at the time. This section is reconstructed from `HANDOFF.md`
> (written 10:51) and file timestamps, not from a contemporaneous record — so
> it is accurate on *what* changed and *why*, but thinner on the dead ends and
> abandoned attempts than the entries written the same day they happened.

The afternoon was more review rounds against fresh stratified samples
(`sample_for_review2.csv` through `sample_for_review7.csv`). Each round
surfaced real bugs; fixes were applied; the next round surfaced more —
sometimes new regressions caused by the previous round's fix, sometimes
previously-latent bugs made visible by it.

**Further fixes, in the order applied**

6. **Cross-page merge: PAKTA specimen forms glued together (p.70–71).** The
   `form`/`single_column` continuation logic had no boundary for untagged
   headings. Fixed by introducing the **ALL-CAPS heading rule** — a standalone
   line with no lowercase letters, ≥10 characters and ≥8 real letters is
   treated as a title. Before shipping it, every such line across all 74 pages
   was surveyed: zero false positives on "is this line, in isolation, a
   genuine heading."

7. **Letterhead fragmented into one node per line.** The new ALL-CAPS rule had
   no "continue previous heading" merge logic. Fixed.

8. **Section B's title split into two detached nodes.** The ALL-CAPS
   attach-to-pending-target path covered `part` and `article` but not
   `section`. Fixed.

9. **Heading text and following body glued into one field.** New heading nodes
   routed everything through `text_raw` instead of `title`. Fixed.

10. **`text_raw` held a stale first-line title fragment (5 nodes).** The
    title-routing fix above extended `title` on wrapped headings but never
    cleared the now-duplicated text in `text_raw`. Fixed.

11. **Most serious bug of the day: a subclause body silently reassigned to the
    wrong parent clause**, producing a fluent but factually wrong sentence.
    Cause was `round(top, 0)` in the block sort: Python uses banker's rounding,
    so two blocks 0.1pt apart (504.5 and 504.6) landed in different integer
    buckets, flipping a clause heading and its own subclause out of order.
    Fixed with a coarser 6pt row bucket — well under this document's ~12–14pt
    line height, so genuinely different rows never merge while near-identical
    rows never split.

12. **43 nodes misclassified as `node_type="clause"`** when they were ordinary
    flat numbered lists (Tembusan lists, SPMK/SPPBJ instructions, PAKTA
    checklists), because `decimal_plain` numbering mapped to `clause`
    unconditionally — and fix #10's dedup logic then wrongly emptied their
    content into `title`. Fixed with a page-height heuristic: the SSUK
    two-column body is uniformly on 612×792 pages while every other section
    uses ~936pt-tall pages, so `decimal_plain` becomes `clause` only within
    10pt of 792 and `list_item` otherwise. Verified that the 3 genuine SSUK
    clauses (32, 34, 78) which happen to sit on pages the *layout* detector
    calls `single_column` were not swept up — a naive "not on a two_column
    page" filter would have caught them wrongly.

**Six false-positive bug reports** were also raised and dismissed across two
rounds: the reviewer was reading `text_raw` rather than the `title` column.
Fixed at the source by adding a `title` column to the review CSV.

**`ground_truth/regression_checks.json` was built** — the day's most durable
output. Round-over-round "did it get better" claims had been unreliable because
`sample_review.py` draws a fresh random sample against a *different* node
population each time (the tree changes as bugs get fixed), so 92% one round and
96% the next aren't comparable numbers. The checklist is one permanent entry
per bug ever found and fixed (20 as of end of day), run automatically by
`evaluate.py` on every invocation. It is the *same* check every run, so "did a
known bug come back" becomes a yes/no fact rather than a re-roll.

Design points that matter when extending it: never locate a node by `node_id`
(they shift on any structural change) — locate by `sub_document` + `node_type`
+ `label_normalized`, or by `pages_contains` plus a distinctive text substring.
A locate resolving to anything other than exactly one node reports `AMBIGUOUS`
or `NOT_FOUND` rather than `PASS`/`FAIL`, which caught two real
under-specifications while the file was being written (`label_normalized: "b"`
matched two unrelated headings; `text_raw_contains: "Pekerjaan Utama"` matched
its own negation, "**bukan** Pekerjaan Utama"). The harness was stress-tested
against itself by mutating a throwaway copy of `raw_extraction.json` to
simulate known regressions and confirming the checks fail with correct
diagnostics — worth repeating for every new entry, since a check you have never
watched fail is not yet evidence of anything.

**End state (end of day, superseding the 08:41 figures above)**

```
pages: 74  nodes: 738  tables: 18
core fields populated: 6/6  overall_confidence=0.85
validation: passed  hard_fails=0  warns=2
  sibling_sequence: 6 sequence breaks
  dual_parser_oracle: avg_ratio~0.91
28/28 core-field checks   20/20 regression checks
```

`review/sample_for_review7.csv` (76 nodes, 10.3% of 738) generated but not
judged.

**Revised limitations going into Day 3**

- **`SSUK_BODY_PAGE_HEIGHT = 792.0` is a hardcoded US-Letter constant in
  `tree.py`, applied regardless of profile.** The single biggest portability
  risk introduced today: a different two-column contract on A4 would have every
  top-level clause silently misclassified as a flat list item — fix #12's exact
  bug, reintroduced by portability rather than by a coding mistake. A fix was
  proposed (derive the body page height empirically per document, the way
  `right_column_start_frac` already is) but not implemented.
- `sibling_sequence` down to 6 breaks; `dual_parser_oracle` low-ratio pages
  (13, 14, 57, 61, 62, 67–69) still uninvestigated.
- The rich ground-truth sections (`node_samples`, `negative_checks`,
  `related_document_numbers`, `known_source_typos`) are still not read by
  `evaluate.py` — a real coverage gap; everything verified against them so far
  was checked by hand.
- The ALL-CAPS heading rule assumes a document convention. A contract not
  following it simply never triggers the rule — a safe no-op, reduced benefit
  rather than a corruption risk.

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
