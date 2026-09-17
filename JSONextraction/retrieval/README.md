# Retrieval

Turns the extraction pipeline's `<pdf-stem>_raw.json` into a searchable vector
collection, scores retrieval against a fixed query set, and optionally answers
questions over the retrieved clauses. For how the pieces fit together and why,
see [`../ARCHITECTURE.md`](../ARCHITECTURE.md#retrieval-retrieval); this file is
how to run it.

```
<pdf-stem>_raw.json
   │  build_embedding_view.py        1 row per tree node + 1 per table row
   ▼
<pdf-stem>_embedding_view.json
   │  load.py  → Mistral mistral-embed (1024-dim)
   ▼
ChromaDB  contracts__mistral-embed__v2_1_0__hnsw-m64ef400
   │  retrievers.py   dense | bm25 | hybrid (RRF)
   ├─► retrieval_evaluate.py   the regression gate
   └─► ask.py  → chat.py        optional answer synthesis
```

The stage never imports from `pipeline/`, and `pipeline/` never imports from
it. Only `<pdf-stem>_raw.json` crosses the boundary.

## Setup

Run everything from `JSONextraction/`, with the project venv.

```powershell
venv\Scripts\python.exe -m pip install -r requirements-retrieval.txt
Copy-Item retrieval\.env.example retrieval\.env
# then paste a key from https://console.mistral.ai/ into MISTRAL_API_KEY
```

`requirements-retrieval.txt` is separate from `requirements.txt` on purpose:
extraction carries no dependency on the retrieval stack. Chroma runs embedded —
no server, no Docker.

### What `retrieval/.env` controls

| Setting | Default | Notes |
|---|---|---|
| `MISTRAL_API_KEY` | — | Needed only for runs that call Mistral — see *When a key is needed* |
| `EMBEDDING_MODEL` | — (required) | Pinned, never defaulted. Part of the collection name |
| `EMBEDDING_BATCH_SIZE` | `64` | Rows per embedding request |
| `EMBEDDING_REQUEST_DELAY` | `0.3` | Seconds before each request; raise it before the batch size if 429s persist |
| `CHAT_MODEL` | empty | Only for `ask --synthesizer mistral`. `open-mistral-nemo` on a free account |
| `CHROMA_DB_PATH` | `./chroma_data` | A relative path resolves from `JSONextraction/`, not from the current directory |
| `CHROMA_COLLECTION_PREFIX` | `contracts` | The rest of the collection name is derived |

### When a key is needed

| Run | API key |
|---|---|
| Unit tests | Never — every model client is faked |
| `build_embedding_view` | Never |
| `load --dry-run` | Never |
| `retrieval_evaluate --retriever bm25`, `ask --retriever bm25` | Never |
| `load`, and `--retriever dense`/`hybrid`/`brute`/`hybrid-brute` | Yes (one embedding call per query) |
| `ask --synthesizer mistral` | Yes, plus `CHAT_MODEL` |

### Secrets and generated data

`retrieval/.env` and every `chroma_data/` directory are gitignored at any depth;
`.env.example` is the only env file tracked, and a unit test fails if that ever
changes or if the template gains a key value. The API key is never logged —
`Settings` excludes it from its `repr`; use `settings.redacted_key` if one has
to appear in output. Chroma's anonymous telemetry is switched off when the
`retrieval` package is imported, so no command or test runs with it on.

## Build and load

```powershell
# 1. project each raw file into an embedding view
Get-ChildItem output\raw\*_raw.json | ForEach-Object {
  venv\Scripts\python.exe -m retrieval.build_embedding_view $_.FullName --out output\embedding
}

# 2. embed and load (resumable — safe to re-run)
$views = Get-ChildItem output\embedding\*_embedding_view.json | % { $_.FullName }
venv\Scripts\python.exe -m retrieval.load $views --dry-run    # what it would send, 0 tokens
venv\Scripts\python.exe -m retrieval.load $views
```

Use the `*_raw.json` glob, not `*.json`: `output\raw\` still holds files under
pre-2026-09-10 names carrying pre-fix values.

> **`load` never deletes.** It adds and updates the ids its views produce.
> An extraction change that *removes* an id — splitting a node changes its text
> and therefore its `embedding_id` — leaves the old row in Chroma, still
> retrievable, defeating the very fix that split it. The `paren_digit_both`
> change orphaned **107** rows, including the merged nodes it replaced. After
> any extraction change, diff the collection's ids against the views and delete
> the difference before re-scoring.

**Resuming.** A row counts as done only if the collection holds its id, so
re-running after a crash, a Ctrl+C or a failed batch embeds only what is
missing. A transient failure (a 429 or 5xx that outlasts six retries, a timeout)
is logged with the affected row ids and the run continues; anything else — a bad
key, a malformed request, a vector-width change, a Chroma write error — stops
the run at that batch, because every later batch would fail the same way.

**After a schema bump that keeps ids stable**, copy vectors instead of paying
for them again:

```powershell
venv\Scripts\python.exe -m retrieval.load $views --reuse-from contracts__mistral-embed__v2_0_0__hnsw-m64ef400
```

Only ids present in the new views are copied, so rows whose text changed are
re-embedded and the stale versions are left behind. The source must be built by
the same embedding model, and its stored text must match the view byte for byte;
either mismatch is refused. The 2.0.0 → 2.1.0 load reused 3,940 vectors and
embedded 582 rows: 36,158 tokens instead of ~243k.

**Changing HNSW parameters** needs a new collection (Chroma fixes them at
creation). Bump `config.INDEX_TAG`, then rebuild from the stored vectors with no
embedding calls:

```powershell
venv\Scripts\python.exe -m retrieval.reindex --from <old-collection> --verify
```

`--verify` reports recall@5 by distance against an exact scan; anything below
1.000 means the index is losing true neighbours.

### What a row is

| | Tree row | Table row |
|---|---|---|
| Source | one node of `structure[]` | one row of `tables[]` |
| `node_type` | `clause`, `subclause`, `list_item`, … | `table_row` |
| `hierarchy_path` | label chain, e.g. `B/33/33.8` | `[table_id, row]`, e.g. `t_062_0/0`; `h` for an emitted header row |
| `text` | title + body, whitespace-collapsed | non-empty cells joined by ` \| ` |
| Cross-references | — | `refs` → Chroma metadata `ref_targets` (`general_terms:A/4/4.1;…`, `?:raw` if unresolved) |

A view reports `structure_row_count`, `table_row_count` and
`table_rows_skipped_empty` separately, so row-count parity with the raw file is
checkable per source. Blank table rows (unfilled template grids) are skipped:
Mistral returns a real vector even for an empty string.

Current corpus (6 specimens, schema 2.1.0):

| Specimen | Rows | Tree | Table | Blank skipped |
|---|---|---|---|---|
| Rancangan Kontrak | 821 | 754 | 67 | 3 |
| kontrakJasa | 1159 | 1016 | 143 | 75 |
| pembangunanRumah | 950 | 849 | 101 | 3 |
| pembangunanSayap | 804 | 725 | 79 | 7 |
| polres | 48 | 48 | 0 | 0 |
| rehabGedung | 834 | 746 | 88 | 3 |
| **Total** | **4616** | **4138** | **478** | **91** |

Around 42% of rows are distinct text — all six specimens are the same standard form.

## The regression gate

```powershell
venv\Scripts\python.exe -m retrieval.retrieval_evaluate                     # hybrid, against the baseline
venv\Scripts\python.exe -m retrieval.retrieval_evaluate --retriever dense --verbose
venv\Scripts\python.exe -m retrieval.retrieval_evaluate --strict            # every query must pass
```

Scores `ground_truth/retrieval_queries.json` (20 queries, k=5) against the
configured collection and compares the result with
`ground_truth/retrieval_baseline.json`, which records the expected passes per
retriever, tokenizer and k.

- **Exit 0**: no query that the baseline passes has started failing.
- **Exit 1**: at least one `[REGRESSION]`. A `[NEW PASS]` is reported but never
  fails the run.
- A baseline recorded on a **different collection** does not apply: the gate
  warns and falls back to the strict verdict, because a model, schema or index
  change is a different system. So does `--max-rank`, which a top-k baseline
  cannot vouch for.

| Flag | Effect |
|---|---|
| `--retriever` | `hybrid` (default), `dense`, `bm25`, or the exact-search ceilings `brute` / `hybrid-brute` |
| `--tokenizer` | `plain` (default), `nostop`, `stem` — bm25 and hybrid only |
| `-k` | Top-k (default: the query set's `default_k`, 5) |
| `--max-rank N` | Also fail a hit found below rank N |
| `--strict` | Ignore the baseline |
| `--update-baseline` | Record this run's passes for this configuration |
| `--verbose` | Print the top-k for passing queries too |

**Recording a baseline is a decision, not a chore.** Run `--update-baseline`
only once a score change has been explained — which change caused it, and why.
Write the explanation into the baseline file's `notes`. Never edit
`retrieval_queries.json` to make the gate pass; change the system and re-run.

### Expected results

Collection `contracts__mistral-embed__v2_1_0__hnsw-m64ef400`, 4616 rows,
recorded 2026-09-16:

| Retriever | Score | Fails |
|---|---|---|
| `hybrid` (default) | **17/20** | q01, q04, q08 |
| `bm25` | 14/20 | q01, q04, q06, q08, q15, q17 |
| `dense` | 13/20 | q01, q02, q04, q10, q13, q18, q20 |
| `hybrid-brute` (ceiling) | 17/20 | q01, q04, q08 |
| `brute` (ceiling) | 13/20 | same as dense |

q01 and q04 fail in every arm, including both exact-search ceilings, so they
are ranking weakness rather than index loss (on the 2.0.0 corpus their best
acceptable rows ranked 20th and 37th).

**`bm25` has lost two queries to corpus growth, and both are the same effect.**
q08 went when the 478 table rows were added; q15 went when `paren_digit_both`
split the SSUK 70.3 `(1)..(6)` list into six short `list_item`s, putting five
copies of `70.3/6` (14.89) above the expected `Pasal 4/1` (12.03), which fell to
rank 8. Each was isolated by rebuilding BM25 without the new rows — the expected
row returns to the top in both cases — so neither is a ranking defect in the
retriever: more, shorter rows shift BM25's length normalisation and IDF for rows
nobody touched. Both are accepted trade-offs recorded in the baseline's `notes`,
not tuned away. **Hybrid is unaffected by both**, which is the robustness
argument for it as the default.

Hybrid and its exact ceiling now agree at 17/20; the ceiling used to score one
lower because q13 passed only at rank 4 on the indexed arm.

### How a query is scored

A query names a target and passes if any row in the top-k answers it.

**Clause target** (`hierarchy_path` given) — accepted rows, reported by kind:

- `exact` — the clause itself, in any specimen;
- `descendant` — any sub-clause beneath it (an ancestor does not count, and
  `C/6` never matches `C/61`);
- `equivalent` — a row whose text is byte-identical to an accepted row. Text of
  60+ characters qualifies on its own; shorter text also has to sit under the
  target clause once a leading section letter is stripped, so generic fragments
  like "Pengadilan." cannot pass for an unrelated clause.

**Content target** (no `hierarchy_path`, `text_contains` given) — every row in
`sub_document` containing the string. Used for table rows, whose path is a
positional table id that differs per specimen.

Optional narrowing on either kind:

- `node_type` — e.g. `table_row`, so the SSUK clause sharing a topic cannot
  answer a question about the SSKK value;
- `text_contains` on a clause target — the row must also carry that exact
  string. Byte-exact, never normalised: this is how identifier and typo survival
  are checked through retrieval;
- `expect_ref` — `{sub_document, path_suffix}`: an accepted hit's `ref_targets`
  must include that clause. The suffix matches on a segment boundary, so `4/4.1`
  accepts `A/4/4.1` and the letterless `4/4.1`.

```json
{
  "id": "q17_sskk_korespondensi",
  "query": "alamat korespondensi para pihak",
  "expect": { "sub_document": "special_terms", "node_type": "table_row", "text_contains": "| Korespondensi |" },
  "expect_ref": { "sub_document": "general_terms", "path_suffix": "4/4.1" },
  "expect_documents": 5,
  "why": "..."
}
```

`expect_documents` records how many specimens hold an exact target. A unit test
checks it against the embedding views on disk, so a stale expectation fails at
test time rather than as a mysterious retrieval miss.

**Adding a query**: pick the target by reading the source content, then write a
query a user would plausibly type. Never run a query first and bless what comes
back — the gate would then measure its own output.

## Asking a question

```powershell
# retrieval only — no model call, no tokens (the default)
venv\Scripts\python.exe -m retrieval.ask "kewajiban penyedia mengasuransikan pekerjaan" --verbose

# lexical only — needs no API key at all
venv\Scripts\python.exe -m retrieval.ask "masa pemeliharaan" --retriever bm25

# with answer synthesis (needs CHAT_MODEL)
venv\Scripts\python.exe -m retrieval.ask "berapa lama masa pemeliharaan?" --synthesizer mistral
```

Identical clauses are collapsed before they are shown or prompted, with the copy
count disclosed (`x5 identik`): a top-5 is often one sentence five times. The
synthesis prompt is extractive — answer only from the supplied clauses, cite
them, say so when they do not contain the answer. If synthesis fails (a rate
limit, an outage), `ask` prints the retrieved clauses and exits 1 instead of
losing them to a traceback.

### Citations

Every source reaches the model with an identifier a reader can look up, and the
prompt forbids building a citation out of numbers found inside the clause text.
That rule is not theoretical — both failures below were observed live:

| Row | Cites as | Why not the obvious thing |
|---|---|---|
| SSUK clause | `Pasal 55.2` | — |
| Surat Perjanjian ayat | `Pasal 5 ayat (2)` | Its label is the bare ordinal `2`, so `Pasal 2` would name a different provision. Cited that way, the model reported that no Pasal 5 ayat (2) had been supplied *while holding its text* |
| SSKK table row | `SSKK hal. 62 (mengacu SSUK 27.1)` | Its path is the positional `t_062_0/3`, meaningless outside this codebase. Given that, the model cited `[27.1]` — a number read out of the row's own first cell. Right by luck there; on a row whose first cell is a price or a date the same behaviour invents a citation |

The cross-reference in a table row's citation comes from its resolved
`ref_targets`, so it is checkable rather than inferred. Unresolved references
(`?:raw`) are dropped: citing one would assert a link the source does not make.

### Asking about one contract

By default a question is answered from all six specimens at once. That is the
right default for *"what does this clause family say"* and the wrong one for
*"what does THIS contract say"* — the specimens are one standard form, so the
other five routinely supply the top hits. `--document` scopes the search:

```powershell
venv\Scripts\python.exe -m retrieval.ask --list-documents

venv\Scripts\python.exe -m retrieval.ask "berapa denda keterlambatan?" --document rehabGedung
venv\Scripts\python.exe -m retrieval.ask "keadaan kahar" --document 843225d8        # key prefix
venv\Scripts\python.exe -m retrieval.ask "keadaan kahar" --document "Rancangan Kontrak"
```

The argument is a filename substring, a `document_key` prefix, or a
comma-separated list of either; matching is case-insensitive. An **ambiguous
term is refused**, never resolved to one of the matches — `pembangunan` matches
two specimens, and quietly picking one would produce a confident, cited answer
about the wrong contract. No match and an empty scope are refused the same way,
each listing what is available.

The scope is printed on every scoped run, not only under `--verbose`: a scoped
answer that looks corpus-wide is this flag's dangerous failure mode. It also
reaches the synthesis prompt, so the model is told the clauses come from one
contract and must not generalise them to the others.

Scoping is applied inside each retriever, never to a finished result list, and
`--document` is the only thing that changes: **without it, retrieval is exactly
what it was before scoping existed**, which is what the gate and every recorded
baseline measure.

A side effect worth knowing: scoping largely removes the duplicate-collapsing
waste. Corpus-wide, a `-k 5` collapses to 1–3 distinct clauses because several
specimens hold the same sentence; scoped to one contract it stays at 5.

| Query (bm25, k=5) | Corpus-wide | Scoped |
|---|---|---|
| kewajiban penyedia terkait asuransi | 1 | 5 |
| denda keterlambatan | 2 | 5 |
| masa pemeliharaan | 3 | 5 |
| jaminan pelaksanaan | 2 | 5 |

Document names come from the embedding views in `output\embedding\`; the keys
come from Chroma. So `--list-documents` shows what can actually be searched — a
view built but never loaded does not appear. With the views absent (they are
gitignored and regenerable) scoping still works by `document_key` prefix, just
without names.

## Tests

```powershell
venv\Scripts\python.exe -m unittest discover -s retrieval\tests
```

202 tests, no API key, no tokens, about 35 seconds. Run them and the gate after
any change to `retrieval/*.py`. If a change touches extraction, rebuild the
views and run `retrieval.load --dry-run`: `pending: 0` means every `embedding_id`
still matches; anything else means node text or structure changed upstream, and
the gate needs re-running once those rows are loaded.

| File | Covers |
|---|---|
| `test_build_embedding_view.py` | Row parity per source, typo and identifier survival, id uniqueness and stability, table rows |
| `test_load.py` | Resume after failure, Chroma as source of truth, stopping on non-transient errors, `--reuse-from` |
| `test_embed.py` | Retry policy, bounded retries, vector-width and count invariants — against a fake Mistral client |
| `test_retrieval_evaluate.py` | Scoring rules, tie-order stability, baselines, content targets, `expect_ref`, the shipped query set |
| `test_retrievers.py` | Dense, BM25, hybrid fusion, brute force, tokenizers, and document scoping on every arm — including that an unscoped search is unchanged and that BM25's IDF stays corpus-wide |
| `test_reindex.py` | Copying without re-embedding, recall verification |
| `test_store.py` | Scope resolution: name and key matching, refusing an ambiguous term, missing views, a collection with no `document_key` |
| `test_chat.py` | Duplicate collapsing, prompt rules (single-contract disclosure, citations from headers only), citation shapes for clauses/ayat/table rows, ref-target parsing, the one-way import boundary, interface conformance |
| `test_ask.py` | Key requirements, fallback when synthesis fails, `--document` and `--list-documents` |
| `test_config.py` | Settings, path resolution, telemetry off, gitignore coverage, key redaction |

## Known limitations

- **The match-kind breakdown is not tie-stable, even though pass/fail is.**
  `dense` and `brute` pass exactly the same 13 queries, yet report
  `exact=10, descendant=3` and `equivalent=6, exact=6, descendant=1`: the kind
  is taken from whichever of several byte-identical rows ranks first. So the
  `equivalent` count is *not* a reliable measure of the clause-path
  inconsistency across specimens (`C/55` vs `55`, plus a third, broken shape
  `B/B.5/1.120` in pembangunanSayap). Compare pass sets, not kind counts.
- **The query set is uneven.** Accepted-set sizes range from 111 rows to 2; a
  pass on a broad query is weak evidence. The gate prints each query's accepted
  count — read it.
- **No chunker.** Rows are 1:1 with tree nodes and table rows; long nodes are
  not split and short siblings are not merged. `paren_digit_both` narrowed this
  — the Surat Perjanjian's ayat are now their own nodes rather than one
  700-character blob — but it only reaches nodes carrying explicit numbering.
  A long node with no internal numbering is still one row.
- **Redundancy is unaddressed in retrieval.** Corpus-wide dedup by text was
  measured and rejected (it destroyed the metadata citations and the gate
  depend on); `ask` collapses duplicates for display only. `--document` sidesteps
  it for single-contract questions but does nothing for corpus-wide ones.
- **Scoped retrieval is not gated.** The query set is corpus-wide, so
  `--document` has unit-test coverage but no measured quality baseline: nothing
  says how *well* single-contract retrieval works. Scoped queries in the gate are
  where `expect_documents` would finally earn its keep.
- **BM25 keeps corpus-wide IDF under a scope**, by decision — it narrows the
  candidates, not the statistics, so a scoped score equals its unscoped one and
  the two are comparable. Per-scope IDF is the plausible alternative and is
  unmeasured; treat it as an experiment to run against the gate, not a fix.
- **`mistral-embed` is a moving alias.** The collection name guards against a
  deliberate model swap, not a silent provider-side retrain at the same width.
  A broad unexplained score drop with no local change is the symptom; a full
  re-embed into a new collection is the fix.
- **Unresolved SSKK references** are left unresolved where the source itself
  is inconsistent: pembangunanSayap numbers its SSUK sub-clauses `1.x`
  continuously while its SSKK cites `41.4`; rehabGedung's SSKK cites 33.19 and
  33.22, which its SSUK does not contain.
- **Free-tier chat models**: `mistral-small-latest` and `mistral-medium-latest`
  return 429 permanently on a free account while `open-mistral-nemo`,
  `open-mistral-7b` and `ministral-*` work. A chat 429 is not evidence of an
  exhausted budget.
- **Not built**: reranking, parent expansion, and metadata pre-filtering on
  anything but `document_key` (no filtering by `sub_document`, page or node
  type).
