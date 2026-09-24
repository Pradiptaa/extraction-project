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
   │  load.py  → Ollama bge-m3 (1024-dim, local)
   ▼
ChromaDB  contracts_rel__bge-m3__v2_1_0__hnsw-m64ef400
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
ollama pull bge-m3          # embeddings, 1024-dim
ollama pull qwen2.5:7b      # answer synthesis
```

`requirements-retrieval.txt` is separate from `requirements.txt` on purpose:
extraction carries no dependency on the retrieval stack. Chroma runs embedded —
no server, no Docker.

Every model runs on a local [Ollama](https://ollama.com). There is no API key
and no per-token cost anywhere in this stage; the hosted-Mistral version of it
is on the `master` branch.

### What `retrieval/.env` controls

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | — (required) | Pinned, never defaulted. Part of the collection name |
| `EMBEDDING_BATCH_SIZE` | `64` | Rows per embedding request |
| `CHAT_MODEL` | empty | Only for `ask --synthesizer ollama` |
| `OLLAMA_HOST` | `http://localhost:11434` | Where the models are served |
| `QUERY_EMBED_ON_CPU` | `true` | Keeps query embedding off the GPU so the chat model stays resident. Bulk loading always uses the GPU |
| `OLLAMA_NUM_CTX` | `8192` | Context window. **Never leave this to Ollama**, whose own default is 2048 and which drops the overflow silently |
| `CHROMA_DB_PATH` | `./chroma_data` | A relative path resolves from `JSONextraction/`, not from the current directory |
| `CHROMA_COLLECTION_PREFIX` | `contracts` | The rest of the collection name is derived |

### What needs Ollama running

| Run | Needs Ollama |
|---|---|
| Unit tests | Never — every model client is faked |
| `build_embedding_view` | Never |
| `load --dry-run` | Never |
| `retrieval_evaluate --retriever bm25`, `ask --retriever bm25` | Never (but the collection must exist) |
| `load`, and `--retriever dense`/`hybrid`/`brute`/`hybrid-brute` | Yes — one embedding call per query |
| `ask --synthesizer ollama` | Yes, plus `CHAT_MODEL` |

A model that has not been pulled fails with the exact `ollama pull` command to
run, rather than a bare HTTP 404.

### Fitting the models in VRAM

`bge-m3` (~1.6 GB) and `qwen2.5:7b` (~5.8 GB) do not fit together in 6 GB. Left
to share the GPU they evict each other on **every** question, and a `hybrid`
answer costs ~20 s, of which ~17 s is the two models reloading.

So a *query* is embedded on the CPU (`QUERY_EMBED_ON_CPU`, on by default).
Measured on this corpus:

| | GPU | CPU |
|---|---|---|
| Embed one query | 110 ms | 125 ms |
| Embed 4686 rows | 1.1 min | 3.1 min |

Fifteen milliseconds buys the chat model an uncontested GPU, and both stay
resident: `ollama ps` shows `bge-m3` at 0.0 GB alongside `qwen2.5:7b` at 4.9 GB.
A `hybrid` answer settles at **~10 s**, the rest being ~2 s of process start-up
and ~3.5 s of actual generation — neither of which model placement can fix.

**Bulk loading keeps the GPU**, where it is 2.8x faster; `load.py` asks for no
placement at all and lets Ollama choose. Turn `QUERY_EMBED_ON_CPU=false` on a
card that holds both models at once.

`--retriever bm25` still calls no embedder, which is worth one point on the gate
(14/20 against 15/20) and saves the embedding step entirely.

### Secrets and generated data

This branch holds no credential of any kind: `Settings` carries none, and a unit
test fails if `.env.example` ever gains a field named like a key, token or
password. `retrieval/.env` and every `chroma_data/` directory stay gitignored at
any depth regardless. Chroma's anonymous telemetry is switched off when the
`retrieval` package is imported, so no command or test runs with it on.

## Build and load

```powershell
# 1. project each raw file into an embedding view
Get-ChildItem output\raw\*_raw.json | ForEach-Object {
  venv\Scripts\python.exe -m retrieval.build_embedding_view $_.FullName --out output\embedding
}

# 2. embed and load (resumable — safe to re-run)
$views = Get-ChildItem output\embedding\*_embedding_view.json | % { $_.FullName }
venv\Scripts\python.exe -m retrieval.load $views --dry-run    # what it would send, nothing embedded
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
missing. A transient failure (a 5xx or a connection error that outlasts six retries, a
timeout) is logged with the affected row ids and the run continues; anything
else — an unpulled model, a malformed request, a vector-width change, a Chroma
write error — stops the run at that batch, because every later batch would fail
the same way.

**After a schema bump that keeps ids stable**, copy vectors instead of
recomputing them:

```powershell
venv\Scripts\python.exe -m retrieval.load $views --reuse-from contracts_rel__bge-m3__v2_1_0__hnsw-m64ef400
```

Only ids present in the new views are copied, so rows whose text changed are
re-embedded and the stale versions are left behind. The source must be built by
the same embedding model, and its stored text must match the view byte for byte;
either mismatch is refused. The 2.0.0 → 2.1.0 load reused 3,940 vectors and
embedded 582 rows. Locally this saves time rather than money: a full re-embed of
all 4,686 rows takes about 80 seconds.

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
an embedding model returns a real vector even for an empty string.

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

Collection `contracts_rel__bge-m3__v2_1_0__hnsw-m64ef400`, 4686 rows,
recorded 2026-09-23:

| Retriever | Score |
|---|---|
| `hybrid` (default) | **15/20** |
| `bm25` | 14/20 |
| `dense` | 11/20 |
| `hybrid-brute` (ceiling) | 15/20 |
| `brute` (ceiling) | 11/20 |

`brute` equals `dense` exactly, so the index is not losing neighbours and the
dense score is the embedding model's own. Against the previous hosted
`mistral-embed` collection, `hybrid` is unchanged at 15/20 while `dense` is two
lower — BM25 covers the difference in the fused arm.

Read small differences carefully: much of this corpus is duplicate text, so
equal-scoring rows are abundant and a collection loaded in a different row order
moves a boundary query in or out of the top-5 on tie order alone. `bm25` is
model-independent and still moved by one point between two collections holding
byte-identical text. Treat ±1 as noise.

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
# retrieval only — no model call (the default)
venv\Scripts\python.exe -m retrieval.ask "kewajiban penyedia mengasuransikan pekerjaan" --verbose

# lexical only — no embedder, so nothing is loaded into VRAM
venv\Scripts\python.exe -m retrieval.ask "masa pemeliharaan" --retriever bm25

# with answer synthesis (needs CHAT_MODEL and a running Ollama)
venv\Scripts\python.exe -m retrieval.ask "berapa lama masa pemeliharaan?" --synthesizer ollama
```

Identical clauses are collapsed before they are shown or prompted, with the copy
count disclosed (`x5 identik`): a top-5 is often one sentence five times. The
synthesis prompt is extractive — answer only from the supplied clauses, cite
them, say so when they do not contain the answer. If synthesis fails (a rate
limit, an outage), `ask` prints the retrieved clauses and exits 1 instead of
losing them to a traceback.

### Quick lookup for core fields

Questions about a contract's core fields are answered straight from
`output\raw\*_raw.json`, with no search and no model call:

| Asks for | Example | Reads |
|---|---|---|
| Contract name | `nama kontrak` | `core.contract_name` |
| Contract number | `nomor kontrak?` | `core.contract_number` |
| Parties | `siapa penyedia?`, `para pihak` | `core.parties` |
| Dates | `tanggal penting kontrak` | `core.key_dates` (day precision only) |
| Numbers | `berapa nilai kontrak?`, `berapa lama masa pemeliharaan?`, `angka penting` | `core.key_numbers` (classified entries only) |

```powershell
venv\Scripts\python.exe -m retrieval.ask "nomor kontrak?" --document polres
venv\Scripts\python.exe -m retrieval.ask "masa pemeliharaan" --route search   # skip lookup
```

`retrieval\lookup.py` routes with fixed rules and sends anything doubtful to
search. That includes a question containing a clause word (`kewajiban`, `jika`,
`asuransi`, `alamat`, …), one over 12 words, and a bare topic with no quantity
word (`masa pemeliharaan` searches; `berapa lama masa pemeliharaan?` looks up).
Without `--document`, every contract gets its own line. A template blank is
reported as *tidak terisi*, not as a miss. If no contract in scope has a value,
the run says so and falls through to search. `--route lookup` never falls
through and refuses questions that aren't about core fields.

Lookup doesn't touch the retrievers, so the gate and its baselines are
unaffected. Known weakness: `key_dates` are unclassified, and extraction reads
some budget codes as dates (`1.03.10` → 2010-03-01). Each date is shown with its
source text for that reason.

### Questions naming a clause

A row's address lives in its label metadata, not in its text, so neither dense
nor lexical search can use it: on this corpus the clause a question named by
number ranked **23rd to 176th** scoped to its own contract, well outside any
top-k. `retrieval\references.py` reads the address, finds the rows whose labels
match, and pins them ahead of the search result.

```powershell
venv\Scripts\python.exe -m retrieval.ask "Berapa lama Masa Pemeliharaan menurut Pasal 5 ayat (3)" --document polres
venv\Scripts\python.exe -m retrieval.ask "Pasal 5 ayat 3" --document polres   # citation only: no search, no key
```

| Understood | Examples |
|---|---|
| Level words, with abbreviations | `pasal`/`psl.`/`ps`, `ayat`/`ay`, `angka`, `butir`, `poin`, `huruf`, `bab`, `bagian`, `klausul`, plus `article`/`section`/`clause`/`paragraph` |
| Number forms | `5`, `33.8`, `1.2.3`, `(3)`, `[3]`, `a`, roman `IX` |
| Part hints | `SSUK`, `SSKK`, `Surat Perjanjian`, `Syarat-Syarat Umum/Khusus Kontrak`, `Lampiran A` — and a part name can stand in for the level word (`SSUK 33.8`) |

A level word is always required, so amounts, durations, dates and regulation
numbers are never read as an address. A lone `huruf b` is relative to a clause
the question never names, so it is ignored.

Matching is on the **end** of the label path, so organisational prefixes a
reader never types (section letters, books) are optional, and split vs joined
numbering is equivalent (`12.4` matches `Pasal 12/4`). Two tiers, best only:

- **tier 0** — the cited level word sits on the segment it addresses. This is
  how `Pasal 5/3` beats an unrelated recital list also numbered `5/3`.
- **tier 1** — the numbers agree but no label carries the level word, which is
  every SSUK path (`B/33/33.8`, `33/33.8`).

Behaviour worth knowing:

- **A cited heading is expanded to its children.** "Pasal 5" alone is a title
  ("MASA KONTRAK"); its ayat carry the provision.
- **A named part narrows the match, but never hides a hit.** If nothing matches
  inside it, the match elsewhere is used and the run says so — extraction can
  file a unit under the wrong part (pembangunanSayap's `Pasal 5/*` sit in
  `general_terms`).
- **No match is reported, never substituted.** `Pasal 33.8` has no matching
  label in pembangunanSayap, whose source numbers the SSUK `1.x`; the run prints
  *tidak ditemukan sebagai label* and searches normally.
- **The search runs on the question minus the address**, filling the remaining
  slots; identical copies collapse as usual, and no row appears twice.
- A citation-only question makes **no model call of any kind**, so it needs no
  model loaded even on `hybrid`.
- `--verbose` prints the address, the tier, how many rows were pinned, and the
  query the search actually ran.

This runs before the retrievers and never inside them, so **retrieval and every
recorded baseline are unchanged** — verified by re-running the gate.

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
scoping is the only thing that changes: **unscoped, retrieval is exactly what it
was before scoping existed**, which is what the gate and every recorded baseline
measure.

### Naming the document in the question

The flag is optional: a question that names a contract is scoped to it.

```powershell
venv\Scripts\python.exe -m retrieval.ask "Pada file Rancangan Kontrak, siapa saja pihak yang terlibat?"
venv\Scripts\python.exe -m retrieval.ask "di dokumen rehabGedung berapa masa pemeliharaan?"
venv\Scripts\python.exe -m retrieval.ask "rehabGedung.pdf berapa lama masa pemeliharaan?"
```

The name is matched on letters and digits only, so `pembangunan rumah` finds
`pembangunanRumah.pdf`, and it is **removed from the query before retrieval** —
left in, "pada file Rancangan Kontrak" is words every specimen contains, and the
scope itself would drive the ranking. `--document` always wins when both are
given.

Which words count as naming a file is deliberately narrow, because scoping to
the wrong contract is what this feature exists to prevent:

| Cue | On a name that matches nothing |
|---|---|
| `file`, `berkas`, an explicit `.pdf` | **Reported.** Nobody writes these by accident, so answering from all six would answer a different question |
| `dokumen`, `kontrak` | **Ignored.** Ordinary nouns here — "penyedia memutuskan kontrak secara sepihak" names no file |

An ambiguous name is refused under both, listing the candidates, exactly as
`--document` does. `test_documents.py` replays the gate's own 20 queries through
the parser and asserts that none of them is scoped or stopped.

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

317 tests, no network and no model of any kind, about 40 seconds. Run them and the gate after
any change to `retrieval/*.py`. If a change touches extraction, rebuild the
views and run `retrieval.load --dry-run`: `pending: 0` means every `embedding_id`
still matches; anything else means node text or structure changed upstream, and
the gate needs re-running once those rows are loaded.

| File | Covers |
|---|---|
| `test_build_embedding_view.py` | Row parity per source, typo and identifier survival, id uniqueness and stability, table rows |
| `test_load.py` | Resume after failure, Chroma as source of truth, stopping on non-transient errors, `--reuse-from` |
| `test_embed.py` | Retry policy, bounded retries, vector-width and count invariants, refusing silent truncation, naming the `ollama pull` fix — against a faked HTTP boundary |
| `test_retrieval_evaluate.py` | Scoring rules, tie-order stability, baselines, content targets, `expect_ref`, the shipped query set |
| `test_retrievers.py` | Dense, BM25, hybrid fusion, brute force, tokenizers, and document scoping on every arm — including that an unscoped search is unchanged and that BM25's IDF stays corpus-wide |
| `test_reindex.py` | Copying without re-embedding, recall verification |
| `test_store.py` | Scope resolution: name and key matching, refusing an ambiguous term, missing views, a collection with no `document_key` |
| `test_chat.py` | Duplicate collapsing, prompt rules (single-contract disclosure, citations from headers only), citation shapes for clauses/ayat/table rows, ref-target parsing, the one-way import boundary, interface conformance |
| `test_ask.py` | Settings loading, fallback when synthesis fails, `--document`, the document inferred from a question, `--list-documents` and `--route` |
| `test_documents.py` | Reading the contract a question names: cue strength, ambiguity, stripping the name from the query, and that the gate's 20 queries are untouched |
| `test_lookup.py` | Routing core-field vs clause questions, template blanks, dedup, number and date filtering, raw file loading |
| `test_references.py` | Reading an address out of a question, label normalisation, both match tiers, heading expansion, relaxed parts, merge order — on invented label shapes, not this corpus's |
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
- **An Ollama tag is a moving target too.** `bge-m3` means whatever was pulled;
  `ollama pull` on another machine may fetch a different build at the same
  width, and the collection name cannot see that. A broad unexplained score drop
  with no local change is the symptom; a full re-embed into a new collection is
  the fix. Pin by digest if this ever bites.
- **Unresolved SSKK references** are left unresolved where the source itself
  is inconsistent: pembangunanSayap numbers its SSUK sub-clauses `1.x`
  continuously while its SSKK cites `41.4`; rehabGedung's SSKK cites 33.19 and
  33.22, which its SSUK does not contain.
- **Model size shows up as fabrication, not as worse prose.**
  `qwen2.5:3b-instruct` answered a denda question by inventing a contract value
  and a clause number that appeared in no retrieved clause; `qwen2.5:7b`, on the
  same clauses, quoted the rate and cited the header it was given. Both are
  fluent. Judge a chat model here by running `--synthesizer null` first and
  checking every figure in the answer against the clauses it was shown.
- **Not built**: automatic validation that a cited clause was actually
  supplied, reranking, parent expansion, and metadata pre-filtering on
  anything but `document_key` (no filtering by `sub_document`, page or node
  type).
