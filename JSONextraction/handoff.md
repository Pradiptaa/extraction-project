# Handoff — contract PDF extraction pipeline

Written 2026-09-10 for continuation in a new chat session. Read this before
touching anything; it explains why the code looks the way it does, not just
what it does.

## 0. Uncommitted state — do this first

**Nothing described below has been committed.** Run `git status` immediately
in a new session — do not assume the working tree matches any commit. As of
writing:

```
 M .gitignore
 M JSONextraction/ARCHITECTURE.md
 M JSONextraction/ground_truth/regression_checks.json
 D JSONextraction/pdfs/4.-Rancangan-Kontrak.pdf                                  (renamed, see below)
 D JSONextraction/pdfs/4._RANCANGAN_KONTRAK_pembangunan_gedung_sayap_direktorat.pdf
 D "JSONextraction/pdfs/RANCANGAN KONTRAK POLRES.pdf"
 D JSONextraction/pdfs/RTJK200.pdf
 D "JSONextraction/pdfs/Rancangan Kontrak Jasa.pdf"
 M JSONextraction/pipeline/core_fields.py
 M JSONextraction/pipeline/evaluate.py
 M JSONextraction/pipeline/main.py
 M JSONextraction/pipeline/numbering.py
 M JSONextraction/profiles/perpres16_konstruksi_v1.json
?? JSONextraction/ground_truth/kontrakJasa.ground_truth.json
?? JSONextraction/ground_truth/pembangunanRumah.ground_truth.json
?? JSONextraction/ground_truth/pembangunanSayap.ground_truth.json
?? JSONextraction/ground_truth/polres.ground_truth.json
?? JSONextraction/ground_truth/rehabGedung.ground_truth.json
?? JSONextraction/pdfs/kontrakJasa.pdf
?? JSONextraction/pdfs/pembangunanRumah.pdf
?? JSONextraction/pdfs/pembangunanSayap.pdf
?? JSONextraction/pdfs/polres.pdf
?? JSONextraction/pdfs/rehabGedung.pdf
?? JSONextraction/requirements-retrieval.txt
?? JSONextraction/retrieval/
```

The user renamed the 5 new specimen PDFs partway through this work (their
original scratch names are still referenced in some `_notes` prose inside the
ground truth files — that's fine, those are historical narration, not paths
the code reads). Current filename mapping:

| Current filename | Original name used during investigation | Ground truth file |
|---|---|---|
| `pdfs/polres.pdf` | `RANCANGAN KONTRAK POLRES.pdf` | `ground_truth/polres.ground_truth.json` |
| `pdfs/rehabGedung.pdf` | `4.-Rancangan-Kontrak.pdf` | `ground_truth/rehabGedung.ground_truth.json` |
| `pdfs/pembangunanSayap.pdf` | `4._RANCANGAN_KONTRAK_pembangunan_gedung_sayap_direktorat.pdf` | `ground_truth/pembangunanSayap.ground_truth.json` |
| `pdfs/pembangunanRumah.pdf` | `RTJK200.pdf` | `ground_truth/pembangunanRumah.ground_truth.json` |
| `pdfs/kontrakJasa.pdf` | `Rancangan Kontrak Jasa.pdf` | `ground_truth/kontrakJasa.ground_truth.json` |
| `pdfs/Rancangan Kontrak.pdf` | (unchanged, the original ground-truth PDF) | `ground_truth/rancangan_kontrak1.ground_truth.json` |

**Before doing anything else**, decide with the user whether to commit this
work (it's a large, coherent, already-verified change — see §3 for the
verification commands) or keep iterating first.

## 1. What this project is

`JSONextraction/` extracts Indonesian government construction contract PDFs
(Perpres 16/2018 standard form: Surat Perjanjian + SSUK + SSKK + Lampiran A/B)
into a structured `raw_extraction.json`, then reduces that to a small
`clean_extraction.json` for search/storage. See `README.md` and
`ARCHITECTURE.md` for the full file-by-file map — don't duplicate that here.
Two things ARE new since those docs were last fully accurate in spirit (the
prose is current, but skim §4 below for what changed underneath it):

- `retrieval/` is a brand new package (see §5) — the start of turning
  `raw_extraction.json` into an embedding view for a vector store.
- The pipeline just went through a generalization pass: tested against 5 new
  real-world specimen PDFs (not just the original one), found and fixed 5
  real bugs, and built independently-verified ground truth for all 5.

## 2. Session narrative — what happened and why it matters

This session (and the one before it) worked through, in order:

1. **Node-ID stability** — confirmed `node_id` (in `pipeline/schema.py`) is a
   plain sequential counter, stable run-to-run on identical input (verified
   with 3 independent runs) but positional, not a durable cross-version
   identity. Pinned with two regression checks (`node_id_stable_first_node`,
   `node_id_stable_deep_node` in `ground_truth/regression_checks.json`) and a
   `pipeline/evaluate.py` addition (`node_id_equals` expect-predicate).
2. **`retrieval/` scaffold** — new package, `requirements-retrieval.txt`,
   `.env`/`.env.example` (gitignored — `.env` is now in root `.gitignore`),
   `EMBEDDING_SCHEMA_VERSION` constant, `build_embedding_view.py` (logging,
   not print), 3 unit tests against a committed synthetic fixture (since real
   `raw_extraction.json` files live in gitignored `output/`).
3. **RAKE as default keyword backend** (was YAKE) — `keywords/extractor.py`,
   `keywords/clean_json.py`. Both backends still work via `--method`.
4. **`output/`/`output_ocr/` reorganized** into `raw/`, `clean/`, `log/`
   subfolders (still gitignored — this is a personal/local convention, not
   something the CLI itself enforces).
5. **Fixed the original "page-height" clause-classification bug**: `tree.py`
   used to decide a `decimal_plain` numbering ("1.") was a real SSUK "clause"
   node by checking if the physical page was 612×792pt (US Letter) — true
   only for the ONE sample PDF this was tuned against. Replaced with a check
   against the profile-declared sub-document scope
   (`expected_invariants.clause_sequence_scope`, e.g. `general_terms`) —
   computed from `main.py`'s `assign_sub_documents`, which now runs BEFORE
   `build_tree` instead of after. This also surfaced and fixed a second bug:
   `assign_sub_documents` used to walk the profile's marker list in strict
   declared order and could never revisit an earlier marker once a later one
   matched; some real specimens bind annexes before the SSUK body, the
   reverse of the declared order. Fixed by taking each marker's first
   occurrence independently, ordered by actual page number.
6. **Built ground truth for the 5 new specimen PDFs**, reading each one
   independently via `pdfplumber` (never trusting the pipeline's own output —
   see the `_verification_method` field in every ground truth file). This is
   what actually validates correctness; the earlier "does it crash" checks
   from the initial generalization pass never did.
7. **Found and fixed 5 more real bugs**, all in `pipeline/core_fields.py`,
   `pipeline/main.py`, `pipeline/numbering.py`, and
   `profiles/perpres16_konstruksi_v1.json` — see §4, the meat of this
   handoff.

## 3. How to verify everything right now

There is no test runner script — each ground truth file is checked
individually via `pipeline.evaluate`. Do this after ANY change to
`pipeline/*.py` or `profiles/*.json`:

```powershell
cd JSONextraction
$SCRATCH = "$env:TEMP\verify_run"
Get-ChildItem pdfs\*.pdf | ForEach-Object {
    $slug = $_.BaseName -replace '[^A-Za-z0-9_.-]', '_'
    venv\Scripts\python.exe -m pipeline.main $_.FullName --out "$SCRATCH\$slug"
}
```

Then for the original (full core + regression suite):
```powershell
venv\Scripts\python.exe -m pipeline.evaluate "$SCRATCH\Rancangan_Kontrak\raw_extraction.json" `
    --ground-truth ground_truth\rancangan_kontrak1.ground_truth.json
```
Expect **28/28 core checks, 22/22 regression checks, RESULT: PASS**. This is
the load-bearing regression baseline — if this ever drops, something broke.

For each of the 5 new specimens (pass `--regression-checks nonexistent.json`
since the regression checklist is specific to the original PDF's content):
```powershell
venv\Scripts\python.exe -m pipeline.evaluate "$SCRATCH\polres\raw_extraction.json" `
    --ground-truth ground_truth\polres.ground_truth.json --regression-checks nonexistent.json
```
Repeat for `rehabGedung`, `pembangunanSayap`, `pembangunanRumah`, `kontrakJasa`
(slug names come from the PowerShell loop above, but note the two-word PDF
names collapse differently — check the loop's actual output paths, or just
match by `$SCRATCH\<pdf-basename-with-invalid-chars-stripped>`).

**Expected current state**: `rancangan_kontrak1`, `polres`, `rehabGedung`,
`pembangunanSayap` all PASS at 100%. `pembangunanRumah` and `kontrakJasa`
FAIL with exactly 2-3 checks each — this is expected and documented (§4.7).
If any OTHER check fails, or if one of the 4 currently-100% files stops
passing, that's a real regression — investigate before doing anything else.

For `retrieval/`:
```powershell
venv\Scripts\python.exe -m unittest discover -s retrieval\tests -v
```
Expect 3/3 pass.

## 4. The 5 bugs fixed this session — what, where, why safe

Every fix below was verified against all 6 ground-truth files (the original
PDF + 5 new specimens) before being accepted, specifically to catch fixes
that helped one document while breaking another. Read `pipeline/evaluate.py`
if you need to know exactly which fields are programmatically checked vs.
purely documentary (`_verification`, `_notes`, etc. are NOT read by any
code) — grep for `gt\[` / `gt\.get` in `evaluate_core`/`main`.

### 4.1 Contract number mis-capture (`core_fields.py::resolve_contract_number`)

**Symptom**: `contract_number` resolved to a stray nearby word ("Kontrak",
"faksimili") instead of `null`, on every specimen with the pattern
`"Nomor : ........................ [diisi nomor Kontrak]"`.

**Root cause**: `_label_lookup`'s regex made the colon optional
(`\s*:?\s*`), so the generic label "Nomor" (in `LABEL_DICTIONARIES`) matched
the WORD "nomor" anywhere in ordinary prose — including inside bracket
drafting instructions like `"[diisi nomor Kontrak]"`, which have no colon
between "nomor" and the next word.

**Fix**: added a `require_colon` parameter to `_label_lookup` (default
`False`, so `contract_name`'s lookup — which genuinely has colon-less
title-block instances — is unaffected), and pass `require_colon=True` only
for `contract_number`. Also added a placeholder-dots filter
(`_PLACEHOLDER_DOTS_RE`, `\.{3,}`) since one specimen's cover-sheet number
has real segments but blank "..." ones embedded (`602.1/.../SP-KONT/...`) —
that's a template, not an assigned number.

**A second false-positive this exposed**: with the bracket-instruction match
gone, a DIFFERENT real "Nomor:" (a law/decree citation in the recitals list,
e.g. `"Peraturan Presiden Nomor 16 Tahun 2018 tentang ..."`) became the new
top-scored candidate. Fixed by rejecting any candidate immediately followed
by `"tentang"` (`_CITATION_TENTANG_RE`) — the standard Indonesian
legal-citation shape, never used for the contract's own number.

**Known remaining gap** (`pembangunanRumah.pdf` specifically): a THIRD
distinct false-positive pattern — `"Kontrak ini dibiayai dari DIPA APBN ...
Nomor : 152.03.DW.7165.RAI.029.054.A.526111."` — a real colon, not a
`tentang`-citation, but still not this contract's own number (it's the
funding/budget-line reference). Left unfixed; see §4.7.

### 4.2 Spurious 3rd party (`core_fields.py::resolve_parties`)

**Symptom**: every specimen produced an extra party with role
`pekerjaan_konstruksi` (all fields null).

**Root cause**: `resolve_parties` treats every `"... selanjutnya disebut
\"X\""` match as a party definition, except terms in
`_SELF_REFERENCE_TERMS`. That set didn't include "pekerjaan konstruksi",
which the standard preamble text also defines this way (`"...melaksanakan
Pekerjaan Konstruksi ... selanjutnya disebut \"Pekerjaan Konstruksi\""`) —
a work/object definition, not a party.

**Fix**: added `"pekerjaan konstruksi"` to `_SELF_REFERENCE_TERMS`.

**Gotcha found while fixing this**: the actual matched text often has an
embedded newline from PDF line-wrapping (`"Pekerjaan\nKonstruksi"`), so the
membership check needs `re.sub(r"\s+", " ", ...)` before lowering/comparing
— a straight `.strip().lower()` doesn't normalize internal whitespace and
silently failed to match the set entry on the first attempt.

### 4.3 Unresolved representative name (`core_fields.py::_extract_party_from_disebut`)

**Symptom**: `representative.name` resolved to `null` despite the name
appearing verbatim in the source, even though the adjacent NIP resolved
fine.

**Two independent causes, both needed fixing**:
- `_NAME_LABEL_RE` / `_POSITION_LABEL_RE` had no `re.IGNORECASE` flag (unlike
  `_ADDRESS_LABEL_RE`, which already did). One specimen (`polres.pdf`) uses
  ALL-CAPS labels (`"NAMA :"`, `"JABATAN :"`) instead of Title-Case
  (`"Nama :"`) — the case-sensitive regex simply never matched.
- The backward search window (`m.start() - 600` chars back from the
  `"selanjutnya disebut"` match) was too short. Measured the real gap
  (Nama-label to its own disebut clause) across specimens: up to 714 chars
  when a long SK/decree citation sits between them. Widened to 950
  (`_PARTY_WINDOW_BACK`), and — since the window no longer respected
  `prev_boundary` the way the org-search window already did — added that
  floor to prevent a wider window from reaching back into the PREVIOUS
  party's own Nama/Jabatan when two parties sit close together.

**Cosmetic bug this exposed**: `_clean_window_value`'s blanket
`.strip(" \t.:")` was eating the trailing period off multi-part
abbreviations (`"S.T., M.T."` → `"S.T., M.T"`) once names started actually
resolving. Fixed with `_ABBREVIATION_TAIL_RE` — only strip a trailing "."
when it's NOT the closing dot of an `X.Y.`-shaped abbreviation.

### 4.4 Marker false positives / missed headings (`main.py`, `profiles/perpres16_konstruksi_v1.json`)

**This was the highest-impact fix** — it affected `general_terms` boundary
placement (hence clause counts), `special_terms` detection (hence
`sub_document_count`), and `annex_a` boundary placement, across most of the
5 new specimens.

**Root cause, part 1**: `assign_sub_documents` matched markers
case-INsensitively. Real section headings in this document family are
ALWAYS printed ALL-CAPS (`"SYARAT-SYARAT UMUM KONTRAK"`,
`"LAMPIRAN A SYARAT-SYARAT KHUSUS KONTRAK"`). The exact same phrases also
turn up in ordinary Title-Case prose wherever a long sentence happens to
line-wrap onto a fresh line starting with those words — e.g. `"...pekerjaan
yang belum tercantum dalam Lampiran A SSKK;"` genuinely starts a new line at
`"Lampiran A SSKK;"` in the PDF's rendering. Case-insensitive matching
couldn't tell these apart; case-sensitive matching cleanly can (verified:
checked case-sensitive vs. case-insensitive match counts for every marker
across all 6 ground-truth specimens — the case-sensitive count exactly
equals the true heading count in every single case, zero exceptions).

**Fix, part 1**: removed `re.IGNORECASE` from the `re.search(...)` call in
`assign_sub_documents` (kept `re.MULTILINE`).

**Root cause, part 2**: several specimens' real `general_terms`/
`special_terms` headings are prefixed with a roman-numeral section number
(`"II. SYARAT-SYARAT UMUM KONTRAK"`, `"III. SYARAT-SYARAT KHUSUS
KONTRAK"`) — the original sample PDF's own heading has no such prefix,
which is presumably why this was never noticed before. The anchored
`^SYARAT-SYARAT UMUM KONTRAK` pattern doesn't tolerate a prefix.

**Fix, part 2**: profile patterns for `general_terms` and `special_terms`
changed to `^(?:[IVX]+\.\s*)?SYARAT-SYARAT (UMUM|KHUSUS) KONTRAK`.

**Important downstream consequence — ground truth files needed correcting,
not just the code**: after this fix, 4 of the 5 specimens' `pages[].sub_document`
sequences now include a genuine `special_terms` section that 3 of the
corresponding ground truth files had marked as "never fires" — that claim
was only ever checked against the OLD, prefix-blind pattern; nobody had
actually looked for a prefixed variant. `rehabGedung`, `pembangunanSayap`,
`pembangunanRumah`, and `kontrakJasa`'s `structural_invariants.sub_document_count`
were all updated from 4 → 5 (with a `_verification` note explaining the
correction) after directly reading the newly-revealed pages and confirming
they contain real, substantive SSKK correspondence-table content (not just a
stray banner). **If you see a `sub_document_count` mismatch in the future,
re-read the actual page content before assuming the code is wrong — it might
be the ground truth that's stale.**

### 4.5 Unscoped `article_word` matching (`pipeline/numbering.py`)

**Symptom**: two specimens (`pembangunanRumah`, `kontrakJasa`) produced an
extra spurious `article` node labelled `"PASAL 1266"`.

**Root cause**: the standard force-majeure/termination clause cites `"Pasal
1266 dan 1267 Kitab Undang-Undang Hukum Perdata"` (the Indonesian Civil
Code) — extremely common boilerplate. In these two specimens' page layout,
that citation happens to wrap onto its own line starting with `"Pasal
1266..."`. `article_word`'s regex used an unbounded `\d+` for the Pasal
number, with no scoping guard at all (unlike `decimal_plain`, which IS
scoped by sub-document since the earlier clause-classification fix).

**Fix**: bounded the number to `\d{1,3}` (max 999) — no real Surat
Perjanjian in this document family has anywhere near that many Pasal
articles, so this cleanly excludes the 4-digit citation without any
realistic risk to genuine headings.

### 4.6 Verification discipline used throughout

For every fix above, before accepting it: extracted each PDF's raw text
independently via `pdfplumber.extract_text()` (see the "how ground truth was
built" note in any `ground_truth/*.ground_truth.json`'s `_verification_method`
field), confirmed the true value/boundary/count by direct inspection or a
freshly-written regex script (never reusing the pipeline's own logic to
"check" the pipeline), THEN checked whether the code's output matched. This
is why every ground truth file's `_verification` notes read as narration —
they're not decoration, they're the actual audit trail for why each expected
value is right.

### 4.7 Known remaining issues — intentionally NOT fixed, documented in the GT files

Two issues remain, in `pembangunanRumah.ground_truth.json` and
`kontrakJasa.ground_truth.json` (see their `_known_bug_*` fields):

1. **`pembangunanRumah`'s contract_number** still resolves to a real but
   wrong value (a DIPA/budget-line reference number, not the contract's own
   number) — see §4.1. Distinguishing "this contract's Nomor:" from "some
   OTHER cited document's Nomor:" generally needs the PRECEDING context
   ("DIPA ... Nomor:" vs. "Surat Perjanjian ini ... Nomor:"), not just what
   follows the value like the `tentang`-filter does. Not attempted — risk of
   an ever-growing pile of narrow special cases outweighed the benefit for
   one specimen.
2. **`pembangunanRumah` and `kontrakJasa` both report 4 parties instead of
   2.** Both documents bundle TWO full "Surat Perjanjian" copies back to
   back — a real, partly-populated one and a blank alternate "KSO"
   (consortium bidder) specimen form of the same document. `resolve_parties`
   currently treats every `"selanjutnya disebut"` match in the whole
   document as belonging to one contract, so it finds the 2-role pattern
   twice. Fixing this properly needs document-structure awareness (detecting
   the second "SURAT PERJANJIAN" copy and stopping party extraction there,
   or being told about it) — a bigger change than a quick pattern fix, not
   attempted this session.
3. `pembangunanRumah` also has a still-unresolved `key_dates` case (a KPA
   decree date phrased `"Nomor: 45 tanggal 01 Juli tahun 2025"` — the "tahun"
   before the year isn't handled by whatever date regex is currently in
   `resolve_key_dates`). Not investigated in depth — flagged, not fixed.

None of these three block anything; they're just the honest remainder.

## 5. `retrieval/` — what exists, what doesn't

```
retrieval/
  __init__.py
  schema.py                 EMBEDDING_SCHEMA_VERSION="1.0.0", embedding_id() helper
  build_embedding_view.py   CLI: reads raw_extraction.json, writes embedding_view.json
  tests/
    __init__.py
    fixtures/sample_raw_extraction.json   committed synthetic fixture (4 nodes)
    test_build_embedding_view.py          3 unittest cases
```

`embedding_id()` derives a key from `sub_document + hierarchy_path (node
"path") + label_normalized` — deliberately NOT `node_id`, because `node_id`
is a positional sequential counter (see `pipeline/schema.py`'s
`NodeIdGenerator` and the two `node_id_stable_*` regression checks) that
shifts for every node after any upstream insertion/deletion, even nodes
whose own content didn't change. Anything needing a durable cross-version
key (a Chroma vector ID, this embedding view) must not use raw `node_id`.

`build_embedding_view.py` produces one row per node in `structure[]` — a 1:1
projection, no chunking/splitting/merging. That's deliberate: this is prep
for a chunker, not the chunker itself. `retrieval/tests/` checks node-count
parity against the source `raw_extraction.json`, plus that a known typo and
a known identifier survive unmodified into the embedded text.

**What does NOT exist yet**: the actual chunker (splitting/merging node text
into retrieval-sized pieces), anything that calls an embedding model, and
anything that touches Chroma. `requirements-retrieval.txt` lists
`python-dotenv` and `chromadb` as forward-looking dependencies for that next
phase — neither is imported by any code yet.

## 6. Things a new session should NOT re-litigate

- **Don't re-add YAKE as the default keyword backend.** RAKE is the default
  per explicit user instruction; YAKE is still available via `--method yake`.
- **Don't move `output/`/`output_ocr/` back to flat files**, and don't
  "fix" the CLI to enforce the `raw/clean/log` subfolder split — that's a
  personal organizational convention for batch-testing multiple PDFs, not a
  pipeline feature. The CLI's own `--out` behavior (writes flat
  `raw_extraction.json`/`clean_extraction.json` into whatever directory you
  point it at) is unchanged and correct.
- **Don't treat `node_id` differences between runs as a bug without first
  checking whether the input or code actually changed** — it's proven
  stable for fixed (code, input) pairs; a difference means something
  upstream changed, which is expected, not a regression in `node_id` itself.
- **Don't add new marker regex special-cases without checking case-
  sensitivity and roman-numeral-prefix tolerance first** — §4.4's fix is
  general-purpose; a NEW false positive is more likely to need a similar
  general fix than another one-off filter.
- **Don't skip re-running the full 6-file ground truth suite** after any
  change to `pipeline/core_fields.py`, `pipeline/main.py`,
  `pipeline/tree.py`, `pipeline/numbering.py`, or any `profiles/*.json` — see
  §3. A fix that helps one specimen while quietly breaking another is
  exactly the failure mode this whole session was about catching.

## 7. Suggested next steps (not started, no commitment implied)

In roughly the order they'd naturally come up, but this is not a plan the
user has approved — confirm before starting any of it:

1. Commit this session's work (nothing is committed yet — see §0).
2. Decide whether to chase §4.7's two remaining issues or accept them as
   documented limitations.
3. Start the actual chunker that consumes `retrieval/build_embedding_view.py`'s
   output.
4. Wire up `retrieval/`'s Chroma integration (embedding model choice is still
   an open `EMBEDDING_MODEL=` blank in `retrieval/.env.example`).
