# Handoff — Contract PDF Extraction Pipeline

Context primer for a new chat session. This is a standalone document, separate
from `PROGRESS_LOG.md` (a dated running log the user controls updates to
directly — don't write to it unless asked). Written at the user's explicit
request to compact this session's work for reuse elsewhere.

---

## 1. What this project is

A Python pipeline that extracts a born-digital Indonesian government
construction contract PDF (`Rancangan Kontrak.pdf`, 74 pages) into a
structured `raw_extraction.json` — a generic recursive node tree plus six
guaranteed "core" fields (document type, contract name, contract number,
parties, key dates, key numbers), built per two design docs:
`analisis_pipeline_kontrak.md` (technical plan) and
`skema_json_dan_logika_ekstraksi.md` (generalized schema/extraction logic).

**Explicit scope boundary, stated by the user up front and never crossed:**
no LLM fallback, anywhere. Every core field resolves via regex/heuristic
strategies; an unresolved field is a documented `null`, never a model call.
No OCR engine wired in either (the sample PDF has a complete native text
layer, so it's never needed here) — pages that would need OCR are flagged,
not silently dropped.

---

## 2. Architecture

```
pipeline/
  probe.py        Stage 1 — per-page geometry/font/image signals
  router.py        Stage 2 — native vs. OCR-needed routing (no OCR engine)
  profiles.py       Stage 3 — profile scoring/selection
  layout.py         Stage 4 — per-page layout classification (single_column /
                    two_column / ruled_table / form / mixed / blank)
  blocks.py         Stage 5 — ordered text blocks + ruled-table extraction
  numbering.py       numbering-token recognizer (style-agnostic: decimal_plain,
                    decimal_dotted, letter_upper, letter_dotted, article_word,
                    chapter_word, paren_digit, latin_lower, roman_lower, bullet)
  tree.py           Stage 6 — recursive node tree build (the most-edited file
                    this session — see §4)
  entities.py        Stage 7 — regex/gazetteer entity cascade
  core_fields.py      Stage 8 — core field resolution (Strategies 1-4)
  validate.py        Stage 9 — validation gate (quality block embedded in output)
  normalize.py       Indonesian currency/date/number-word/rate parsing
  schema.py          value-object + ID helpers
  main.py           CLI orchestrator: python -m pipeline.main "<pdf>" --out output
  evaluate.py        scores raw_extraction.json against ground truth (§5)
  sample_review.py    builds a stratified human-review CSV sample
profiles/
  generic_contract_v1.json       mandatory fallback, always populates core
  perpres16_konstruksi_v1.json    matches this document family (Indonesian
                                  govt construction contract, Perpres 16/2018)
ground_truth/
  rancangan_kontrak.ground_truth.json    simple/original ground truth (v1)
  rancangan_kontrak1.ground_truth.json    rich, hand-verified ground truth (v2,
                                          actively used) — has _verification
                                          notes, negative_checks, node_samples,
                                          known_source_typos sections that
                                          evaluate.py does NOT yet read (see §7)
  regression_checks.json         permanent, accumulating per-bug checklist
                                  (§5) — 20 entries as of this writing
review/
  sample_for_review7.csv          latest stratified sample (76 nodes,
                                  10.3% of 738), NOT YET JUDGED
requirements.txt    just pdfplumber
venv/               Windows venv, see README.md for setup
README.md           full setup/run/evaluation instructions (kept up to date
                    throughout — read this for command syntax, not this file)
```

Only `layout.py`/`blocks.py`/`tree.py` do the structural work; profiles do
**not** gate any of it — a profile only supplies `sub_document_markers` (for
tagging which part of the document a page belongs to) and validation
invariants. This matters for §8 (generalization).

---

## 3. Current state (as of last full run)

```
profile: perpres16_konstruksi_v1 (score=1.0)
pages: 74  nodes: 738  tables: 18
core fields populated: 6/6  overall_confidence=0.85
validation: passed  hard_fails=0  warns=2
  sibling_sequence: 6 sequence breaks       ← not investigated, low priority
  dual_parser_oracle: avg_ratio~0.91, several pages <0.9 (13,14,57,61,62,67-69)
                                            ← not investigated

Core-field evaluation:      28/28 PASS
Bug regression checklist:   20/20 PASS  (see §5, §6)
Human review (76-node sample): generated, not yet judged
```

Command to reproduce from scratch:
```bash
cd JSONextraction
python -m pipeline.main "Rancangan Kontrak.pdf" --out output
python -m pipeline.evaluate output/raw_extraction.json \
    --ground-truth ground_truth/rancangan_kontrak1.ground_truth.json
```

---

## 4. The story so far — Day 1 build, then extensive iterative debugging

### Day 1 (separate session; see `PROGRESS_LOG.md` if present)
Built the pipeline end-to-end from the two design docs. Got the two-column
SSUK split working (79-80 clause boundary detection was the central
technical risk called out in the design doc), all 6 core fields resolving,
venv + README + requirements.txt. See `PROGRESS_LOG.md` for that session's
own detailed log if it still exists in the repo.

### This session — iterative review-driven debugging (the bulk of the work)

The user supplied an expanded, hand-verified ground truth
(`rancangan_kontrak1.ground_truth.json`, with per-field `_verification` notes
citing exact page numbers and quotes from the PDF) and did **repeated rounds
of human review** against `sample_review.py`'s stratified CSV samples. Each
round surfaced real bugs; fixes were applied; the next round surfaced more
(sometimes new regressions from the previous round's fix, sometimes
previously-latent bugs newly visible). This iterative loop is the main
substance of the session. Full bug-by-bug checklist:

| # | Bug | Root cause | Status |
|---|---|---|---|
| 1 | `duration` only found first match | `.search()` vs `.finditer()` | Fixed |
| 2 | `monetary` entity type didn't exist | missing feature | Fixed |
| 3-6 | Column bleed: clauses 15/19/28/56 titles bled into body | `_line_groups` merged column-0/column-1 words into one string *before* column split — see detail below | Fixed |
| 7-8 | Table captions/footnotes merged across tables & pages | ruled_table orphan blocks had no isolation/gap logic | Fixed |
| 9 | `penalty_rate` same first-match bug as #1 | same `.search()` mistake, caught via self-audit | Fixed |
| 10 | `evaluate.py` let 2 expectations PASS against 1 real match | test-harness bug, no per-match consumption tracking | Fixed |
| 11-12, 15-16, 18-19 | **6 false-positive bug reports** across 2 rounds | reviewer read `text_raw`, not the `title` column that was added | N/A — added `title` column to review CSV |
| 13 | Cross-page merge: PAKTA specimen forms glued together (p.70-71) | `form`/`single_column` continuation logic had no boundary for untagged headings | Fixed → introduced the ALL-CAPS heading rule |
| 14 | Letterhead fragmented into separate nodes per line | new ALL-CAPS rule had no "continue previous heading" merge logic | Fixed |
| 17 | Section B's title split into 2 detached nodes | ALL-CAPS "attach to pending target" only covered `part`/`article`, not `section` | Fixed |
| 20 | Heading text + following body glued into one field | new heading nodes used `text_raw` for everything, not `title` | Fixed |
| 21 | `text_raw` held a stale first-line title fragment (5 nodes) | title-routing fix extended `title` on wrapped headings but never cleared the duplicate in `text_raw` | Fixed |
| 22 | **Most serious bug found**: subclause body silently reassigned to the wrong parent clause, producing a fluent but factually wrong sentence | `round(top, 0)` sort used Python's banker's-rounding; two blocks 0.1pt apart (504.5 vs 504.6) landed in different integer buckets, flipping a clause heading and its own subclause out of order | Fixed (coarser 6pt row bucket) |
| 23 | 43 nodes misclassified as `node_type="clause"` when they were ordinary flat numbered lists (Tembusan lists, SPMK/SPPBJ instructions, PAKTA checklists) | `decimal_plain` numbering mapped to `"clause"` unconditionally; the dedup fix from #21 then wrongly emptied their content into `title` | Fixed via page-height heuristic (see below) — **verified 3 genuine SSUK clauses (32, 34, 78) were NOT swept up in the fix**, since a naive "not on a two_column page" filter would have wrongly caught them too |

**Key technical fixes, explained** (for whoever picks this up):

- **Column-split root cause (#3-6)**: `_line_groups` clusters words into
  lines by y-proximity *before* column identity is assigned. A short
  left-column heading and the first line of its right-column body routinely
  share the same `top`, so they got concatenated into one string before any
  split could happen. Fixed in `blocks.py` by computing the right column's
  own start position empirically (minimum leading x0 among lines past the
  detected gap — NOT the page's midpoint boundary, which is thrown off by
  hanging-indent formatting) and splitting merged lines there.

- **The ALL-CAPS heading rule** (introduced for #13, refined through #14/17/20):
  a standalone line with no lowercase letters, ≥10 chars, ≥8 real letters is
  treated as a title. Verified by surveying every such line across all 74
  pages before shipping it — zero false positives on "is this line, in
  isolation, a genuine heading." The bugs it caused (#14, #17, #20) all came
  from *interaction* with other in-progress structures (a multi-line
  letterhead, a split section heading, a heading immediately followed by
  body text), not from the core heuristic being wrong.

- **Row-bucket tie-break (#22)**: `round(504.5, 0) == 504` but
  `round(504.6, 0) == 505` in Python (banker's rounding). Any two blocks that
  straddle a `.5` boundary can sort out of order. Fixed with a 6pt bucket
  (`round(top / 6.0)`), well under this document's ~12-14pt line height, so
  genuinely different rows never merge but near-identical rows never split.

- **Page-height clause/list-item classification (#23)**: the SSUK two-column
  body is uniformly on 612×792 (US Letter) pages; every other section of the
  document uses ~936pt-tall pages (documented in the original analysis,
  section A.6, but never wired into the classifier until now).
  `decimal_plain` numbering only becomes `node_type="clause"` when the page
  height is within 10pt of 792; otherwise it becomes `"list_item"`. Verified
  this doesn't misclassify the 3 genuine SSUK clauses that happen to land on
  pages the two-column *layout* detector calls `single_column` (pure body
  continuation, no heading of its own on that specific page) — page height
  is a document-structure signal, independent of per-page layout detection.

---

## 5. The evaluation/regression methodology (built mid-session, important)

Early in the debugging loop, round-over-round "did it get better" claims
were unreliable because `sample_review.py` draws a **fresh random stratified
sample every time**, against a **different node population** each time (the
tree changes as bugs get fixed). A 92% this round and 96% last round aren't
comparable numbers. The user pushed back on this explicitly ("why does every
adjustment surface another bug... it's a tie, not better or worse") — fair
critique, and it led to building:

**`ground_truth/regression_checks.json`** — one permanent, accumulating entry
per bug ever found and fixed (20 entries currently, covering bugs #3-8, 13,
14, 17, 20-23 above). `pipeline/evaluate.py` runs it automatically every
time (default path, or `--regression-checks` to point elsewhere / skip).
Unlike the random sample, this is the *same* check every run — "did a known
bug come back" becomes a yes/no fact instead of a re-roll of the dice.

Design details that matter if extending this file:
- **Never locate by `node_id`** — these are assigned sequentially and shift
  on any structural change. Locate by stable fields instead: `sub_document`
  + `node_type` + `label_normalized`, or `pages_contains` + a distinctive
  `text_raw_contains`/`title_contains`/`text_raw_equals` substring for
  unlabeled nodes (captions, headings).
- **A locate that doesn't resolve to exactly one node reports `AMBIGUOUS` or
  `NOT_FOUND`, distinct from `PASS`/`FAIL`** — both still count as failures,
  but the distinction matters: a check silently validating against the wrong
  node is worse than useless. This caught two real under-specifications
  while the file was being built: `label_normalized: "b"` alone matched two
  unrelated section headings in different sub-documents, and
  `text_raw_contains: "Pekerjaan Utama"` matched its own negation
  ("**bukan** Pekerjaan Utama") too. Both were fixed to exact/scoped locates
  before shipping.
- **The harness was stress-tested against itself**: mutated a throwaway copy
  of `raw_extraction.json` (never touching pipeline code) to simulate
  several known regressions and confirmed the checks actually fail with
  correct diagnostics, then confirmed the real file still passes clean.
  Repeat this whenever extending the checklist with a new entry — don't
  trust a check you haven't watched fail.
- **Maintenance rule**: every time a new bug is found and fixed, add one new
  entry before moving on. This is what keeps the list monotonic. If a
  deliberate future redesign makes an old entry obsolete, remove it
  explicitly with a note — never let an entry silently start failing without
  a decision about why.

`pipeline/evaluate.py` also runs (independently, all in one invocation):
1. Core-field checks against `ground_truth/*.ground_truth.json` (document
   type, contract name/number, parties, dates, key numbers, structural
   invariants, identifier survival).
2. The regression checklist (above).
3. Optionally, `--review-csv <path>` scores a human-judged
   `sample_review.py` output (completeness + accuracy, ≥99% target).

---

## 6. Known limitations (explicit, not swept under the rug)

- **`SSUK_BODY_PAGE_HEIGHT = 792.0` is a hardcoded US-Letter-height
  constant in `tree.py`, applied unconditionally regardless of profile.**
  This is the single biggest portability risk introduced this session — see
  §8. Flagged to the user, not yet fixed (proposal exists, not implemented).
- **Party extraction** (`core_fields.resolve_parties`) only handles two
  marker styles: `PIHAK PERTAMA`/`PIHAK KEDUA`, and `"... atas nama X,
  selanjutnya disebut 'Y'"`. A contract using neither pattern resolves
  `parties` to `[]` with a `no_parties_detected` flag — a real gap, not a
  silent guess.
- **`sub_document_count` is pinned at 5** to match what the profile
  currently declares, but the ground truth's own notes point out the
  document has 8 logically distinct parts — SPPBJ (p.5) and SPMK (p.6)
  carry real extractable metadata (the penalty rate and a second duration
  attestation both live in SPMK) and are currently absorbed into
  `main_agreement`, meaning their content is currently mis-attributed at the
  sub-document level even though the values themselves extract correctly.
- **`sibling_sequence` warnings (6)** and **`dual_parser_oracle` low-ratio
  pages** (13, 14, 57, 61, 62-69) have never been investigated — carried
  forward from very early in the session, still open.
- **`evaluate.py` doesn't read the rich ground truth's `node_samples`,
  `negative_checks`, `related_document_numbers`, or `known_source_typos`
  sections** — all present in `rancangan_kontrak1.ground_truth.json`, none
  wired up. Everything verified against them so far was done by manual
  inspection, not automated checking. Real coverage gap.
- **The ALL-CAPS heading rule assumes a document convention** (title case /
  all-caps = heading). A contract that doesn't follow this convention would
  simply never trigger it (safe no-op, falls back to older/plainer
  behavior) — not a corruption risk, just reduced benefit.
- **No LLM fallback, no OCR** — by explicit design choice, not oversight
  (see §1).
- **Number-word parser** (`normalize.parse_number_words_id`) is correct for
  the hundreds range used by duration cross-validation; not a general
  Indonesian numeral parser (million-scale compounds like "lima ratus juta"
  are out of scope, nothing currently needs them).

---

## 7. Open proposal — not yet implemented

**Generalize `SSUK_BODY_PAGE_HEIGHT` away from a hardcoded constant.** The
user asked directly whether this pipeline would work on a different contract
document with a different format, and the honest answer surfaced this: a
different two-column contract on A4 or any non-Letter page size would have
every top-level clause silently misclassified as a flat list item — the
exact bug just fixed (#23), reintroduced by portability rather than a coding
mistake. Proposed fix: derive the "body page height" empirically per
document, the same way `right_column_start_frac` in `layout.py` was derived
— from whichever page-height cluster is actually dominant among pages
showing a genuine two-column layout signal in *that* document — rather than
assuming 792pt universally. **This was proposed but the user had not yet
said whether to implement it when this handoff was written.**

---

## 8. Generalization assessment (full answer given to the user)

Checked the code directly (not just described intent) before answering
"will this work on other contracts":

**Genuinely general**: the whole architecture, `generic_contract_v1` fallback
(always populates `core` even with a shallow tree), fraction-based column
boundary detection (computed per-document, never a fixed coordinate), the
row-bucket fix, title/text_raw separation design, the entity cascade
pattern, Indonesian legal-vocabulary regexes (dates/currency/NIP-shaped IDs)
— portable across similar Indonesian government/legal documents.

**Safe no-ops on a different document**: `perpres16_konstruksi_v1`'s own
match signals (falls back to generic automatically), `_RUNNING_HEADER_RE`
(literal `"LAMPIRAN"` — inert if absent).

**The one real risk**: `SSUK_BODY_PAGE_HEIGHT` (§7 above) — not gated by
profile, applies to every document via `tree.py` regardless of which
profile matched, and fails *silently wrong* rather than failing safe.

---

## 9. Process notes for whoever continues this

- The user explicitly does **not** want `PROGRESS_LOG.md` written to unless
  they ask — that instruction stands going forward.
- The user values being asked before large/ambiguous fixes are implemented,
  but has repeatedly authorized "audit + fix" once a root cause is clearly
  diagnosed and scoped. Read the room per-message; recent pattern has been:
  diagnose and present findings clearly first, then proceed once confirmed.
- The user pushes back hard (correctly) on overclaiming progress — don't
  say "genuinely better" based on a metric that isn't actually comparable
  round-to-round (see §5's origin story). Be precise about what's confirmed
  vs. hypothesized, and verify hypotheses against actual data before
  proposing fixes (several rounds in this session involved a hypothesis
  being flat wrong on inspection — always confirm before fixing).
- When investigating a bug, prefer reading actual block/coordinate data
  (`pipeline.probe`, `pipeline.layout`, `pipeline.blocks` called directly in
  a Python one-liner) over guessing from symptoms — most root causes in this
  session were only found this way, not by reading the pipeline code in the
  abstract.
