# Contract Extraction → Search → RAG Pipeline
### Analysis & Technical Plan for `Rancangan_Kontrak.pdf`

**Headline finding: this PDF does not need OCR.** It is a born-digital Microsoft Word 2016 export with a complete, tagged text layer on all 74 pages. Introducing OCR would *reduce* accuracy, not improve it.

**Second headline finding: the standard Sastrawi preprocessing recipe you proposed will silently invert the legal meaning of clauses in this contract.** Evidence in section G. This is the single most important item in this document.

---

## A. PDF Analysis

### A.1 Document identity and provenance

| Property | Value |
|---|---|
| Pages | 74 |
| File size | 593,018 bytes (~579 KB) |
| Producer / Creator | Microsoft® Word 2016 |
| Author | GATOT |
| Created | 2023-05-04 · Modified | 2024-08-09 |
| PDF version | 1.7 |
| **Tagged** | **Yes** (structure tree present) |
| Encrypted | No · Forms: none · JavaScript: no |
| Page rotation | 0° on all 74 pages |

~579 KB for 74 pages is roughly 8 KB/page. A scanned page at usable OCR resolution is 100–500 KB. The file size alone rules out a scan.

### A.2 Is OCR required? No.

`pdffonts` lists 23 embedded/referenced TrueType fonts (ArialNarrow, FootlightMTLight, TimesNewRoman, CambriaMath). A scanned PDF lists **no fonts at all**. Every page carries real, selectable text:

```
p  1: chars=  970 words= 131 imgs=2 tables=0
p 30: chars= 1770 words= 220 imgs=0 tables=0
p 60: chars= 1778 words= 222 imgs=0 tables=0
p 73: chars= 1064 words= 167 imgs=0 tables=4
p 74: chars=    2 words=   1 imgs=0 tables=0
```

Total extracted: **201,051 characters** across 74 pages. No page falls to the near-zero character count that signals a raster page. The handful of images (pages 1, 2, 5, 6, 72) are the government letterhead logo and decorative elements, not text-bearing scans.

**Conclusion: use native text extraction. OCR is not merely unnecessary — it is harmful here**, because it would replace a lossless digital text layer with a probabilistic transcription, introducing errors into contract numbers, monetary values, and clause identifiers that are currently perfect.

> Keep an OCR fallback in the codebase, but gate it behind a per-page detector (see F.2). This template will be reused for other contracts, and *those* may genuinely be scans. Route per page, not per document.

### A.3 Document structure

The file is not one document — it is a **compound contract bundle** of five distinct sub-documents with different layouts and independent numbering:

| Pages | Sub-document | Layout | Numbering scheme |
|---|---|---|---|
| 1–6 | **Surat Perjanjian** (cover, parties, recitals, SPMK) | Single column + label:value blocks | `Pasal 1` … `Pasal 5` |
| 7–61 | **SSUK** — Syarat-Syarat Umum Kontrak | **Two-column (borderless table)** | `1.` … `79.` → `1.1`, `1.2` … → `a.`, `b.` → `1)`, `2)` |
| 62–66 | **SSKK** — Syarat-Syarat Khusus Kontrak | **3-column ruled table** | Keyed by *reference* to SSUK (`4.1 & 4.2`, `6.3.b & 6.3.c`) |
| 67–68 | **Lampiran A SSKK** | Ruled data tables | Table rows |
| 69–74 | **Lampiran B SSKK** (RKK — Rencana Keselamatan Konstruksi) | Mixed forms + tables | **Restarts at `- 11 -`, `- 12 -`, `13`** |

**Critical:** Lampiran B has its own page-number footers (`- 11 -`, `- 12 -`, `13`) that disagree with the PDF page index. Any citation system must distinguish *PDF page* from *printed page label*, or users will be sent to the wrong page.

### A.4 The SSUK two-column layout — the main extraction hazard

Pages 7–61 (55 of 74 pages, ~75% of the document) are a **borderless Word table**. Left column = clause heading; right column = clause body. There are **zero ruling lines** on these pages (`rects=0, lines=0`), so `pdfplumber.find_tables()` returns **0 tables** — the structure is purely positional.

With `pdftotext -layout` the structure is visible:

```
23. Rapat Persiapan       23.1   Paling lambat 7 (tujuh) hari kalender sejak
    Pelaksanaan Kontrak          diterbitkannya SPMK dan sebelum pelaksanaan
                                 pekerjaan, ...
```

Without `-layout` (i.e. naive `PdfReader.extract_text()`, the default in most tutorials), **reading order collapses**:

```
37.2

Dalam hal tidak terjadi perubahan kondisi
lapangan seperti yang dimaksud pada pasal 37.1
```

The clause number is severed from its body by a blank line, and heading text from the left column interleaves unpredictably. **A naive extraction of this document produces text that is structurally unusable for RAG.** This is the single biggest technical risk in the project.

**Good news — it is reliably solvable.** I tested coordinate-based column splitting on pages 7–60. Filtering words with `x0 < 110` matching `^\d{1,2}\.$`:

> **79 top-level clause headings found, numbered 1 through 79, strictly sequential, with zero gaps.**

That is a clean result, and it doubles as a permanent validation invariant (see I.3).

### A.5 The SSKK cell-wrap hazard

Pages 62–66 *do* have ruling lines (`p62: rects=88`, `p67: rects=407`), so `pdfplumber` detects them as tables. But `-layout` mode shows column bleed when a middle-column cell wraps:

```
 4.2 & 5.1   Wakil      Sah Wakil Sah Para Pihak sebagai berikut:
             Para Pihak
```

Column 2 (`Wakil Sah Para Pihak`) and column 3 (`Wakil Sah Para Pihak sebagai berikut:`) merge on one visual line. **For pages 62–69 and 72–73, use `pdfplumber`'s ruling-line table extraction into proper cell objects — never flat `-layout` text.**

### A.6 Page geometry is inconsistent

```
PAGE SIZES: {(612, 936): 16, (610, 936): 2, (612, 792): 56}
```

Three different page sizes. The 612×936 pages are the front matter and annexes; 612×792 (US Letter) is the SSUK body. Two pages are 610×936 — a 2pt anomaly.

**Consequence: never hard-code absolute coordinates.** All column thresholds and header/footer margins must be expressed as *fractions of page width/height*, computed per page. A hard-coded `x0 < 110` cutoff — the one that worked in my test on the 612-wide pages — will silently misfire on a 610-wide page. This is exactly the kind of bug that produces a quietly corrupted corpus.

### A.7 Indonesian-specific content characteristics

Non-ASCII character inventory across the whole document:

| Char | Code | Count | Note |
|---|---|---|---|
| `…` | U+2026 | 338 | Ellipsis — **template blanks awaiting fill-in** |
| `”` `“` | U+201D/201C | 20 | Smart quotes around **defined terms** |
| `√` | U+221A | 5 | Checkbox marks in RKK safety forms |
| `–` | U+2013 | 3 | En-dash |
| `‰` | U+2030 | 3 | **Per-mille — a monetary penalty rate** |
| `•` | U+2022 | 3 | Bullet |

Notable characteristics:

1. **This is a draft template, not an executed contract.** 338 ellipses mark unfilled blanks (`Rp. ..................,00`, `[nama wakli Penyedia]`). Bracketed `[...]` instructions to the drafter are interleaved with binding text. **These must be flagged, not silently indexed** — otherwise RAG will confidently answer "the contract value is Rp. .........".

2. **Smart quotes define legal terms.** `selanjutnya disebut "Kontrak"`, `"PPKom"`, `"Penyedia"`. The quotation marks are semantically load-bearing — they mark definition sites. Normalizing them away destroys a useful signal; the SSUK Pasal 1 definitions list (1.1–1.x) is a natural glossary to extract.

3. **Dense abbreviation usage** with the `yang selanjutnya disingkat X` pattern: APIP, HPS, HSP, KSO, SPMK, SPPBJ, SSUK, SSKK, RKK, RMPK, RKPPL, RMLL, SMKK, PPN, DPPA-SKPD, PA/KPA. These are the highest-value search terms in the document and are exactly what a stemmer will mangle.

4. **`‰` is genuine, not an encoding artifact.** Context: `1‰ (satu perseribu) dari harga bagian` — a penalty rate. Note the document expresses the *same* concept two different ways: `1/1000 (satu per seribu)` on page 5 and `1‰ (satu perseribu)` on page ~48. A search for either form must find both.

5. **Indonesian number formatting**: `Rp. 1.500.000.000,00` — period as thousands separator, comma as decimal. Opposite of English convention. Any numeric normalization written with English assumptions will corrupt every monetary figure.

6. **Structured identifiers** that must survive verbatim:
   - Contract no. `08/PUPRPRKP-B.PNK/SP-PPK`
   - DPPA-SKPD `1.03.10.2.01.05`
   - NIP `19810618 200604 2 008` (18-digit civil servant ID with internal spaces)
   - Decree ref. `945/140/2023`

7. **Non-embedded fonts**: `ArialMT`, `Arial-BoldMT`, `TimesNewRomanPSMT`, `Arial-ItalicMT` show `emb: no`. Text extraction is unaffected (WinAnsi encoding is standard), but *page rendering* for visual QA will substitute fonts. Relevant only if you rasterize for review.

8. **Legal citation density**: the recitals cite UU No. 2/2017, UU No. 11/2020, PP No. 22/2020, PP No. 14/2021, Perpres No. 16/2018, Perpres No. 12/2021, Perpres No. 17/2019. Worth extracting as first-class entities.

### A.8 Headers, footers, and page transitions

- Pages 1, 2, 5, 6 carry a repeated 4-line government letterhead (`PEMERINTAH KABUPATEN SUKAMARA` / `DINAS PEKERJAAN UMUM...` / address line). This repeats verbatim and will pollute BM25 term statistics if indexed.
- Lampiran B carries `- 11 -`, `- 12 -`, `13` footers.
- The SSUK body (pages 7–61) appears to have **no running header/footer**, which simplifies things considerably.
- **Clauses split across page boundaries.** Example: clause 23.1 ends on one page and 23.2 begins after the form feed. A clause-level chunker must stitch across the page break rather than emit two fragments.
- Page 74 contains 2 characters (`13`) — effectively blank, and a divide-by-zero risk in any per-page quality metric.

### A.9 Section boundary detectability — verdict

**Yes, reliably**, with sub-document-specific detectors:

| Boundary type | Detection signal | Confidence |
|---|---|---|
| Sub-document splits | Heading regex (`SYARAT-SYARAT UMUM KONTRAK`, `LAMPIRAN A ...`) | High |
| `Pasal N` (pp. 1–6) | `^\s*Pasal\s+\d+\s*$` centered | High |
| SSUK Part (`A. KETENTUAN UMUM`) | `^[A-Z]\.\s+[A-Z ]+$` at left margin | High |
| SSUK clause `N.` | Left column, `x0 < 0.18·width`, bold font | **High — validated 79/79** |
| SSUK sub-clause `N.M` | Right column, `^\d{1,2}\.\d{1,2}$` at column start | High |
| Letter items `a.` `b.` | Indent depth within right column | Medium |
| Roman/paren items `1)` `2)` | Deeper indent | Medium |
| SSKK rows | pdfplumber ruled-table cells | High |
| Signature blocks | `Untuk dan atas nama`, `[tanda tangan]`, `NIP.` | Medium-High |

---

## B. Key Extraction Challenges

Ranked by risk to the finished system:

1. **Sastrawi stopword removal inverts negation.** `tidak` is in Sastrawi's 126-word stoplist. See G.1. **Severity: critical.**
2. **Two-column SSUK reading order** (75% of the document) collapses under naive extraction. **Severity: critical.**
3. **Stemming destroys clause identifiers and contract numbers.** `Pasal 37.2` → `pasal 37 2`. See G.2. **Severity: high.**
4. **Template placeholders indexed as facts** — 338 ellipsis blanks. **Severity: high** (produces confident wrong RAG answers).
5. **Dual page numbering** (PDF index vs. printed label in Lampiran B) breaks citations. **Severity: medium-high.**
6. **Inconsistent page geometry** silently breaks hard-coded coordinate thresholds. **Severity: medium-high.**
7. **SSKK cell-wrap column bleed** merges distinct table cells. **Severity: medium.**
8. **Clauses spanning page breaks** fragment chunks. **Severity: medium.**
9. **Repeated letterhead** skews BM25 IDF. **Severity: low-medium.**
10. **Indonesian decimal convention** (`.` thousands / `,` decimal). **Severity: medium** if any numeric normalization is attempted.

---

## C. Research Findings

### C.1 PDF parser selection

A comparative study across 10 parsers on the DocLayNet dataset covering six document categories found that <cite index="5-1">for text extraction, PyMuPDF and pypdfium generally outperformed the other tools, though all parsers struggled with scientific and patent documents</cite>. The same study evaluated table extraction and found that <cite index="5-1">among rule-based tools, recall was poor outside the Manual and Tender categories, with Camelot achieving the highest score in the Tender category at 0.72, while PyMuPDF showed the most consistent recall across categories</cite>. Your document is a government procurement contract — closest to the *Tender* category, which is one of the two categories where rule-based extraction actually performs well. That is encouraging.

On the speed/fidelity trade-off between the two leading candidates: <cite index="7-1">PyMuPDF extracts plain text roughly 8–12× faster than pdfplumber, but ships under AGPL-3.0, which pushes commercial users toward a paid license or open-source obligations; pdfplumber is MIT-licensed and produces visibly better tables, at the cost of speed</cite>. The same source notes that <cite index="7-1">PyMuPDF can supply bounding boxes for every text block, but its table detection is rudimentary — it returns cells without always grouping them correctly into rows and columns, whereas pdfplumber was designed for tables from the start</cite>.

pdfplumber's own documentation confirms the character-level basis for this: <cite index="4-1">pdfminer.six provides pdfplumber's foundation, focusing on parsing PDFs, analyzing layouts and object positioning; PyMuPDF is substantially faster but requires non-Python MuPDF software</cite>. And the character-level access is precisely what enables coordinate work: <cite index="8-1">pdfplumber gives access to every character's bounding box, font size, weight, and precise page coordinates — the granularity that makes its table extraction accurate, because it reconstructs rows and cells from character positions rather than guessing column boundaries</cite>.

**This maps directly onto your document.** You have exactly two problems that need character coordinates: the borderless two-column SSUK (needs `x0` per word) and the ruled SSKK tables (needs cell reconstruction). pdfplumber is built for both. At 74 pages, its speed disadvantage is irrelevant — the whole document parses in under 5 seconds. And the licensing difference is decisive: **AGPL-3.0 is a genuine constraint for a system handling government contract data**, whereas pdfplumber's MIT license imposes nothing.

The `-layout` mode of `pdftotext` (Poppler) is a useful *cross-check* oracle — it uses an independent C++ implementation, so agreement between it and pdfplumber is meaningful evidence of correctness (see I.2).

### C.2 Preprocessing for legal Indonesian

Sastrawi is the de facto Indonesian stemmer. Its documentation describes it plainly: <cite index="17-1">Sastrawi reduces inflected Indonesian words to their base form; `'Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan'` stems to `'ekonomi indonesia sedang dalam tumbuh yang bangga'`</cite>. The algorithm has a real research pedigree — <cite index="10-1">it adapts the Nazief-Andriani algorithm with modified confix-stripping, drawing on Asian's 2007 RMIT thesis on effective techniques for Indonesian text retrieval and Arifin et al.'s enhanced confix-stripping stemmer</cite>. Note, however, that the upstream PHP repository is <cite index="16-1">marked inactive</cite>, and there is a licensing subtlety worth flagging: <cite index="16-1">Sastrawi's code is MIT-licensed, but its root-word dictionary comes from Kateglo under CC-BY-NC-SA 3.0</cite> — a **non-commercial** clause. If this system is ever commercialized, that dictionary license needs legal review.

Critically, the standard preprocessing literature treats stopword removal as uncontroversial because it is evaluated on *sentiment analysis* — a task where function words genuinely carry little signal. One such study describes the standard recipe: <cite index="9-1">stopword removal targets frequently-occurring words held to carry no meaning, such as "yang", "di", "untuk", and "dari", using the Sastrawi stoplist</cite>. **That justification does not transfer to legal text**, and I verified empirically that it fails badly here — see section G.

### C.3 Retrieval architecture

The evidence for hybrid retrieval is strong and converging. One 2026 practitioner benchmark reports that <cite index="20-1">BM25 handles exact entity matches but misses semantic similarity, while vector search handles conceptual queries but fumbles on SKUs and technical identifiers; running both concurrently and fusing with reciprocal rank fusion moves recall@10 from 65–78% to around 91%, with the fusion step costing about 6ms against 500ms–2s of LLM inference</cite>. RRF is preferred over weighted score blending for a specific reason: <cite index="22-1">it is rank-based, sidestepping the fact that BM25 scores range from zero to potentially infinity while cosine similarity ranges from -1 to 1, which makes normalization fragile</cite>.

For jargon-dense corpora specifically, the guidance is to tilt toward sparse retrieval: <cite index="24-1">for extremely jargon-heavy corpora such as legal contracts, API documentation, and error logs, give sparse search more candidates — for instance retrieval_k=100 for sparse against 30 for dense</cite>. The same source notes a scale effect directly relevant to a 74-page corpus: <cite index="24-1">BM25 tends to be relatively more valuable for smaller corpora, where IDF statistics are meaningful and distinctive, with dense retrieval's semantic understanding becoming comparatively more important as the corpus grows to tens of thousands of chunks</cite>.

Recent academic work on text-and-table documents reinforces this, finding that <cite index="19-1">BM25 outperforms dense retrieval on their benchmark, contextual retrieval provides consistent gains through document-level enrichment, and fusion method choice significantly impacts performance</cite>. Research on RAG over large legal datasets similarly augmented dense retrieval with BM25 <cite index="23-1">to explicitly match salient terms such as party names between queries and documents</cite>.

On chunk sizing, the general default is <cite index="18-1">recursive character splitting at 256–512 tokens, which produces chunks that work well for both sparse and dense retrieval; semantic chunking can improve dense recall but sometimes produces chunks too small or too large for effective BM25 scoring</cite>. But there is an important caveat for structured documents — <cite index="22-1">uniform chunk sizes actually neutralize BM25's document-length normalization, so metadata-enriched chunks carrying section titles and parent context are preferable</cite>. **Your document argues for structure-aware chunking over fixed-size splitting**, because its clause boundaries are already semantically meaningful and machine-detectable.

### C.4 Embeddings for Indonesian

BGE-M3 is the strongest general option: <cite index="25-1">it performs dense, multi-vector, and sparse retrieval simultaneously, supports over 100 languages in a common semantic space, and handles inputs up to 8192 tokens</cite>. Its unified design has a practical architectural payoff — <cite index="34-1">BGE-M3's hybrid mode replaces three separate components (a dense encoder, BM25, and a reranker) with a single model whose outputs feed directly into a hybrid pipeline</cite>.

For Indonesian specifically, LazarusNLP maintains purpose-built models; their collection identifies <cite index="33-1">`all-indo-e5-small-v4` — `intfloat/multilingual-e5-small` fine-tuned on all available supervised Indonesian datasets — as their current best Indonesian sentence embedding model</cite>.

One finding worth internalizing: on long-input retrieval benchmarks, <cite index="27-1">BM25 wins outright at 82.2, since lexical matching is hard to beat when looking for one passage inside a long document</cite>. That is precisely your use case — finding one clause inside a 74-page contract. **Do not treat the vector index as the primary retriever here.**

---

## D. Technology Comparison

### D.1 Extraction engine

| Option | Verdict | Reasoning |
|---|---|---|
| **pdfplumber** | ✅ **Primary** | MIT license. Char-level `x0`/`top` coordinates — required for the borderless SSUK columns. Best-in-class ruled-table extraction for SSKK. 74 pages parses in seconds; speed penalty irrelevant at this scale. |
| PyMuPDF / `pymupdf4llm` | ⚠️ Secondary | Fastest and excellent text quality, but **AGPL-3.0** is a real constraint for government contract data. Rudimentary table grouping. Good as a *cross-check* parser and for page rasterization in QA. |
| `pdftotext -layout` (Poppler) | ✅ **Cross-check oracle** | Independent C++ implementation. Agreement with pdfplumber = strong correctness evidence. Not the primary path (flat text loses cell structure). |
| pypdf / PyPDF2 | ❌ Reject | No shape/coordinate access. Its default extraction is exactly what scrambles the SSUK columns (demonstrated in A.4). |
| Camelot / Tabula | ⚠️ Optional | Camelot scored best in the Tender category. Worth trying on the p67/p73 annex tables *if* pdfplumber underperforms. Tabula needs a JVM. |
| Tesseract / OCRmyPDF | ❌ Not for this file | No text layer to recover — there already is one. **Keep as a per-page fallback for future scanned contracts only.** |
| Unstructured.io | ⚠️ Optional | Broad coverage, good default when document type is unknown. But it abstracts away the coordinate control you specifically need for the two-column layout. |
| Docling / LayoutLMv3 / TATR | ⚠️ Future | Transformer layout models showed better cross-category versatility in the DocLayNet study. Overkill for one well-behaved born-digital file; revisit if you onboard genuinely messy scans. |
| Cloud OCR (Azure DI, Google DocAI, Textract) | ❌ Reject | Strong products, but sending government contract data to a third party is a procurement and data-residency question, not just a technical one. Unnecessary given a perfect local text layer. |

**Recommendation: pdfplumber as primary, `pdftotext -layout` as an independent validation oracle, PyMuPDF for rasterization during QA only.**

### D.2 Indonesian NLP

| Option | Verdict |
|---|---|
| **Sastrawi** | ⚠️ **Use narrowly and with a modified stoplist.** Stemming only for the BM25 field, never for stored text. Its default stoplist must be edited (remove `tidak`, `dapat`, `bukan`, etc.). Note the CC-BY-NC-SA dictionary license. |
| **Custom legal stoplist** | ✅ **Required.** Derive from Sastrawi's 126 words minus all negation/modal/deontic terms. ~40–60 words, hand-reviewed. |
| PySastrawi / go-sastrawi | ➖ Same algorithm, different runtime. No benefit. |
| NLTK / spaCy Indonesian | ⚠️ spaCy has no official `id` pipeline; community models are thin. |
| Stanza (Stanford) | ✅ **Recommended for NER/lemmatization.** Has a trained Indonesian model (UD GSD). Lemmatization is more conservative than confix-stripping — better for legal text. |
| IndoNLU / IndoBERT (IndoLEM) | ✅ Best-in-class Indonesian transformers if you later want a trained clause classifier or NER over party names. Overkill for v1. |
| **Regex + gazetteer** | ✅ **Primary for entities.** Contract numbers, NIP, dates, currency, clause refs, and the `yang selanjutnya disingkat X` abbreviation pattern are all cleanly regex-able and far more reliable than a statistical NER for these forms. |

### D.3 Search backend

| Option | Verdict |
|---|---|
| **PostgreSQL + `pgvector` + `tsvector`** | ✅ **Recommended for v1.** One database holds source-of-truth JSONB, BM25-ish full-text, and vectors. Transactional consistency between the sparse and dense indexes — which directly solves the "stale BM25 index serving fresh vector results" failure mode. Postgres ships an `indonesian` text-search configuration with a Snowball stemmer. |
| Elasticsearch / OpenSearch | ✅ Strongest pure BM25 + native RRF, and a mature Indonesian analyzer. Choose this if you expect many contracts and heavy faceted search. Operationally heavier. |
| Qdrant / Weaviate / Milvus | ⚠️ Excellent vector stores with native hybrid + RRF, but you'd still run a separate relational store for source-of-truth. Two systems to keep in sync. |
| BGE-M3 unified sparse+dense | ⚠️ Elegant (one model, three retrieval modes), but its learned sparse weights are harder to debug than BM25 term matches. For a legal system where *explainability of why a clause matched* matters, plain BM25 is preferable. |
| `rank_bm25` (in-memory Python) | ✅ **Perfect for the prototype.** 74 pages ≈ a few hundred chunks — fits in RAM trivially. Use it in Phase 4 to validate retrieval quality before committing to infrastructure. |

---

## E. Recommended Architecture

```
Rancangan_Kontrak.pdf
   │
   ├─[0] INGEST & FINGERPRINT ─────────────────────────
   │     SHA-256 · pdfinfo · pdffonts · per-page geometry
   │     → manifest.json
   │
   ├─[1] PAGE ROUTER (per-page, not per-document) ──────
   │     chars_per_page > 100 AND fonts_present?
   │       ├─ YES → native extraction  (all 74 pages here)
   │       └─ NO  → OCRmyPDF + Tesseract (--language ind)
   │
   ├─[2] SUB-DOCUMENT SEGMENTER ───────────────────────
   │     Surat Perjanjian | SSUK | SSKK | Lampiran A | Lampiran B
   │
   ├─[3] LAYOUT-SPECIFIC EXTRACTORS ───────────────────
   │     ├─ pp.1–6   single-column + label:value parser
   │     ├─ pp.7–61  TWO-COLUMN SPLITTER (x0 < 0.18·page_width)
   │     ├─ pp.62–66 pdfplumber ruled-table → cells
   │     ├─ pp.67–68 pdfplumber ruled-table → cells
   │     └─ pp.69–74 mixed forms + printed-page-label capture
   │
   ├─[4] HIERARCHY BUILDER ────────────────────────────
   │     Part → Clause → Sub-clause → Item → Sub-item
   │     stitch across page breaks · assign stable clause_id
   │
   ├─[5] ENTITY & PLACEHOLDER TAGGER ──────────────────
   │     regex: contract_no, NIP, currency, dates, clause refs,
   │            abbreviations, legal citations, "…" placeholders
   │
   ├─[6] ► RAW LAYER  raw_extraction.json  ◄ SOURCE OF TRUTH
   │        (immutable · byte-faithful · never overwritten)
   │
   ├─[7] VALIDATION GATE ──────────────────────────────
   │     79-clause invariant · dual-parser diff · coverage ≥99.5%
   │     FAILS → pipeline halts, no downstream write
   │
   ├─[8] DERIVATION LAYER (many views, one source) ─────
   │     ├─ text_display   (light normalization only)
   │     ├─ text_bm25      (stem + custom stoplist + entities re-injected)
   │     ├─ text_embed     (clean prose + contextual header)
   │     └─ entities[]     (structured, typed)
   │
   ├─[9] CHUNKER  clause-aware, 200–450 tokens, parent-linked
   │
   └─[10] DUAL INDEX  Postgres: JSONB + tsvector + pgvector
           → hybrid retrieval (RRF) → rerank → RAG context
```

**Core design principle: the raw layer is immutable and written once.** Everything else is a *derivation* that can be regenerated by re-running a deterministic transform. If you later change your mind about stemming, you rebuild the BM25 field — you never re-parse the PDF, and you never risk the source of truth. This is what makes the "separate extraction from preprocessing" requirement architecturally real rather than just a naming convention.

---

## F. Recommended JSON Schema

### F.1 Design rationale

Your example schema was close but conflated two things. I'm splitting it into **three files** rather than one, because they have different lifecycles:

| File | Lifecycle | Purpose |
|---|---|---|
| `raw_extraction.json` | Write-once, immutable | Legal source of truth, byte-faithful |
| `enriched.json` | Regenerable | Derived text views + entities |
| `chunks.jsonl` | Regenerable, versioned | Retrieval units |

Also changed from your draft:
- **`ocr_confidence` → `extraction_confidence`** with a `method` discriminator. For native extraction there is no OCR confidence; forcing the field invites a meaningless `1.0` that masks real problems. A method-aware confidence is honest.
- **Added `page_label` alongside `page`** — non-negotiable given the Lampiran B renumbering (A.3).
- **Added `is_template_placeholder`** — the 338 ellipses must be machine-visible.
- **Added `char_span`** — offsets into the page's raw text, so any derived value can be traced back to exact source characters.

### F.2 `raw_extraction.json` — source of truth

```jsonc
{
  "schema_version": "1.0.0",
  "document": {
    "doc_id": "kontrak-08-PUPRPRKP-B.PNK-SP-PPK",
    "source_file": "Rancangan_Kontrak.pdf",
    "sha256": "<hash of original bytes>",
    "page_count": 74,
    "language": "id",
    "document_status": "draft_template",   // ← 338 unfilled blanks
    "pdf_metadata": {
      "producer": "Microsoft® Word 2016",
      "author": "GATOT",
      "creation_date": "2023-05-04T02:35:49Z",
      "modification_date": "2024-08-09T09:28:14Z",
      "tagged": true
    },
    "contract_metadata": {
      "contract_number": "08/PUPRPRKP-B.PNK/SP-PPK",
      "contract_type": "Kontrak Harga Satuan",
      "work_package": "Peningkatan Jalan Mekar Desa Natai Sedawak",
      "sub_activity": "PEMBANGUNAN JALAN",
      "dppa_skpd_number": "1.03.10.2.01.05",
      "budget_source": "APBD Kabupaten Sukamara TA 2023",
      "fiscal_year": 2023,
      "contract_value": null,               // ← blank in template
      "duration_days": 120,
      "employer": {
        "organization": "Dinas Pekerjaan Umum dan Penataan Ruang dan Perumahan Rakyat dan Kawasan Permukiman Kabupaten Sukamara",
        "representative": "GIAJENG WULANDARI, ST",
        "nip": "19810618 200604 2 008",
        "role": "Pejabat Pembuat Komitmen Bidang Bina Marga"
      },
      "contractor": { "organization": null, "representative": null }
    },
    "extraction": {
      "pipeline_version": "1.0.0",
      "extracted_at": "<iso8601>",
      "parsers": { "primary": "pdfplumber==0.11.x", "oracle": "poppler-pdftotext" }
    }
  },

  "pages": [
    {
      "page": 30,
      "page_label": null,          // printed footer label, if any
      "width": 612.0, "height": 792.0, "rotation": 0,
      "extraction_method": "native",
      "layout_type": "two_column_ssuk",
      "has_ruling_lines": false,
      "raw_text": "<verbatim pdfplumber output for this page>",
      "char_count": 1770
    }
  ],

  "structure": [
    {
      "node_id": "ssuk.c37.s2",
      "parent_id": "ssuk.c37",
      "path": ["SSUK", "D. PERUBAHAN KONTRAK", "37. Perubahan Kontrak", "37.2"],
      "sub_document": "SSUK",
      "part": "D. PERUBAHAN KONTRAK",
      "clause_number": "37",
      "clause_title": "Perubahan Kontrak",
      "subclause_number": "37.2",
      "level": 2,
      "node_type": "subclause",

      "text_raw": "Dalam hal tidak terjadi perubahan kondisi lapangan seperti yang dimaksud pada pasal 37.1 namun ada perintah perubahan dari Pejabat yang berwenang untuk menandatangani Kontrak, ...",

      "pages": [30],
      "spans_page_break": false,
      "bbox": { "page": 30, "x0": 226.4, "top": 88.1, "x1": 540.2, "bottom": 246.7 },
      "char_span": { "page": 30, "start": 0, "end": 412 },
      "reading_order": 512,

      "extraction_confidence": { "method": "native", "score": 1.0, "flags": [] },
      "is_template_placeholder": false,
      "children": ["ssuk.c37.s2.a", "ssuk.c37.s2.b", "ssuk.c37.s2.c"]
    }
  ],

  "tables": [
    {
      "table_id": "sskk.t1",
      "page": 62,
      "sub_document": "SSKK",
      "caption": "SYARAT-SYARAT KHUSUS KONTRAK",
      "headers": ["Pasal dalam SSUK", "Ketentuan", "Data"],
      "rows": [
        { "cells": ["4.1 & 4.2", "Korespondensi", "Alamat Para Pihak sebagai berikut: ..."],
          "refs_ssuk": ["4.1", "4.2"] }
      ],
      "bbox": { "x0": 54.0, "top": 96.0, "x1": 558.0, "bottom": 720.0 },
      "extraction_method": "pdfplumber_ruled"
    }
  ],

  "artifacts": {
    "headers_footers": [
      { "pages": [1,2,5,6], "text": "PEMERINTAH KABUPATEN SUKAMARA ...", "type": "letterhead" },
      { "pages": [72,73,74], "text": "- 12 -", "type": "page_label" }
    ],
    "images": [ { "page": 1, "bbox": [...], "role": "logo" } ]
  }
}
```

**The `refs_ssuk` field is doing important work.** SSKK rows are keyed by reference to SSUK clauses (`4.1 & 4.2`), so this is a genuine cross-document link. Materializing it means a query about *korespondensi* can retrieve both the general rule (SSUK 4.1) and its specific override (SSKK). Without it, the retrieval system will return the general clause and miss the binding specific one — a serious legal error.

### F.3 `enriched.json` — derived views

```jsonc
{
  "node_id": "ssuk.c37.s2",
  "source_ref": { "file": "raw_extraction.json", "node_id": "ssuk.c37.s2" },

  "text_display": "Dalam hal tidak terjadi perubahan kondisi lapangan seperti yang dimaksud pada pasal 37.1 namun ada perintah perubahan...",
  "text_bm25":    "hal tidak jadi ubah kondisi lapang maksud pasal_37.1 perintah ubah pejabat wenang tandatangan kontrak",
  "text_embed":   "[SSUK › D. Perubahan Kontrak › Pasal 37 Perubahan Kontrak › 37.2] Dalam hal tidak terjadi perubahan kondisi lapangan...",

  "entities": [
    { "type": "clause_ref",   "text": "pasal 37.1",  "normalized": "SSUK.37.1", "char_span": [62, 72] },
    { "type": "party_role",   "text": "Pejabat yang berwenang untuk menandatangani Kontrak",
                              "normalized": "PPKom", "char_span": [140, 191] }
  ],
  "negation_present": true,
  "deontic_modality": "permissive",   // wajib | dilarang | dapat | harus
  "token_count": 68
}
```

`negation_present` and `deontic_modality` are cheap regex-derived fields that pay for themselves. Legal queries are overwhelmingly about *obligation* (`wajib`, `harus`), *prohibition* (`dilarang`, `tidak dapat`), and *permission* (`dapat`). Making modality a filterable field lets you answer "what is the contractor prohibited from doing?" as a metadata query rather than hoping the embedding captures it.

### F.4 `chunks.jsonl` — retrieval units

```jsonc
{
  "chunk_id": "kontrak-08::ssuk.c37.s2::0",
  "doc_id": "kontrak-08-PUPRPRKP-B.PNK-SP-PPK",
  "node_id": "ssuk.c37.s2",
  "parent_node_id": "ssuk.c37",
  "chunk_index": 0, "chunk_total": 1,
  "text": "[Kontrak 08/PUPRPRKP-B.PNK/SP-PPK › SSUK › D. Perubahan Kontrak › 37. Perubahan Kontrak › 37.2]\n\nDalam hal tidak terjadi...",
  "text_bm25": "...",
  "metadata": {
    "sub_document": "SSUK", "part": "D. PERUBAHAN KONTRAK",
    "clause_path": "SSUK/37/37.2", "clause_number": "37", "subclause_number": "37.2",
    "pages": [30], "page_labels": [],
    "node_type": "subclause", "level": 2,
    "deontic_modality": "permissive", "negation_present": true,
    "is_template_placeholder": false,
    "contract_number": "08/PUPRPRKP-B.PNK/SP-PPK",
    "fiscal_year": 2023,
    "prev_node_id": "ssuk.c37.s1", "next_node_id": "ssuk.c37.s3"
  },
  "embedding_model": "BAAI/bge-m3", "embedding_version": "1.0.0"
}
```

---

## G. Recommended Preprocessing Pipeline

### G.1 ⛔ Critical finding: Sastrawi stopword removal inverts legal meaning

I ran Sastrawi's default `StopWordRemover` against a real clause pattern from this contract:

```
IN : Penyedia tidak dapat mengalihkan pekerjaan utama kepada sub penyedia
OUT: Penyedia dapat mengalihkan pekerjaan utama sub penyedia
```

**The input says the contractor MAY NOT subcontract the main work. The output says the contractor MAY.** The meaning is exactly reversed.

The cause: `tidak` (not) is one of the 126 entries in Sastrawi's default stoplist. I verified directly:

```
stopword list size: 126
is "tidak" a stopword? True
is "dapat" a stopword? True
is "bukan" a stopword? False
```

Both the negation particle `tidak` and the modal `dapat` are removed. In sentiment analysis this is a minor loss. In a construction contract governing subcontracting rights, penalties, and termination, **it is a defect that could produce materially wrong legal advice.**

> **Recommendation: never apply Sastrawi's default stoplist to this corpus.** Build a custom stoplist that removes only true function words (`yang`, `di`, `dari`, `pada`, `untuk`, `dengan`, `dan`, `atau`, `adalah`) and explicitly **retains** every negation, modal, and deontic term: `tidak`, `bukan`, `tanpa`, `kecuali`, `dilarang`, `wajib`, `harus`, `dapat`, `boleh`, `selain`, `belum`, `jangan`.

### G.2 ⚠️ Sastrawi stemming destroys legal identifiers

Same test, using the default stemmer:

| Input | Sastrawi output | Damage |
|---|---|---|
| `sesuai Pasal 37.2 dan SSKK 6.3.b` | `sesuai pasal 37 2 dan sskk 6 3 b` | **Clause refs destroyed** — 37.2 becomes two tokens |
| `Nomor : 08/PUPRPRKP-B.PNK/SP-PPK` | `nomor 08 puprprkp-b pnk sp-ppk` | **Contract number shattered** |
| `Rp. 1.500.000.000,00` | `rp 1 500 000 000 00` | **Currency destroyed** — 1.5 billion → 6 meaningless tokens |
| `1‰ (satu perseribu)` | `1 satu ribu` | **`‰` silently dropped** — the rate becomes `1` |
| `sebelum PPN` | `belum ppn` | **"before VAT" → "not yet VAT"** — over-stemming changes meaning |
| `Penyedia` | `sedia` | Defined party term reduced to an unrelated root |
| `Perjanjian` | `janji` | "Agreement" → "promise" |

Three separate failure modes here: (1) punctuation stripping fragments structured identifiers; (2) confix-stripping over-stems defined legal terms; (3) non-ASCII symbols are dropped entirely.

`Penyedia → sedia` is worth dwelling on. `Penyedia` is a **capitalized defined term** — the document says `selanjutnya disebut "Penyedia"`. It is a proper noun in the contract's own terms, and stemming it to `sedia` ("available") both destroys the link to the definition and creates false matches against unrelated uses.

> **Recommendation: apply stemming ONLY to the `text_bm25` field, and only after a protection pass** that masks entities (clause refs, contract numbers, currency, NIP, dates, capitalized defined terms) behind placeholder tokens, stems the remaining prose, then re-injects the originals verbatim. Never stem `text_display` or `text_embed`.

### G.3 Per-field preprocessing matrix

| Step | `text_raw` (truth) | `text_display` | `text_bm25` | `text_embed` | RAG context |
|---|---|---|---|---|---|
| Unicode NFC normalize | ❌ | ✅ | ✅ | ✅ | ✅ |
| De-hyphenate line breaks | ❌ | ✅ | ✅ | ✅ | ✅ |
| Collapse whitespace | ❌ | ✅ | ✅ | ✅ | ✅ |
| Strip letterhead/footers | ❌ | ✅ | ✅ | ✅ | ✅ |
| Lowercasing | ❌ | ❌ | ✅ | ❌¹ | ❌ |
| Punctuation removal | ❌ | ❌ | ⚠️² | ❌ | ❌ |
| **Stopword removal** | ❌ | ❌ | ⚠️³ | ❌ | ❌ |
| **Stemming (Sastrawi)** | ❌ | ❌ | ⚠️³ | ❌ | ❌ |
| Number removal | ❌ | ❌ | **❌ NEVER** | ❌ | ❌ |
| Clause-ID removal | ❌ | ❌ | **❌ NEVER** | ❌ | ❌ |
| Abbreviation expansion | ❌ | ❌ | ➕⁴ | ➕⁴ | ❌ |
| Entity protection | ❌ | ❌ | ✅ required | ✅ | ✅ |
| Contextual header prepend | ❌ | ❌ | ❌ | ✅ | ✅ |
| Smart-quote normalize | ❌ | ⚠️⁵ | ✅ | ✅ | ⚠️⁵ |

¹ Modern transformer embedders are case-aware and trained on cased text; lowercasing loses the signal that `Penyedia` is a defined term.
² Preserve `.` and `,` inside numeric and clause-ref patterns. Strip only sentence-terminal punctuation.
³ **Custom legal stoplist only** (G.1), **after entity protection** (G.2).
⁴ Additive only — index both `SSKK` and `Syarat-Syarat Khusus Kontrak`. Never replace.
⁵ Keep `""` in display and RAG context — they mark definition sites.

### G.4 Entity handling rules

| Entity | Rule |
|---|---|
| **Clause refs** (`Pasal 37.2`, `SSKK 6.3.b`) | Mask → `__CLAUSEREF_0__` → restore. Also index as structured `clause_ref` entities so "what does 37.2 say" is a metadata lookup, not a text search. |
| **Currency** (`Rp. 1.500.000.000,00`) | Parse with Indonesian convention (`.`=thousands, `,`=decimal) into a numeric field. Index both the surface string and the parsed value for range queries. |
| **Percent / per-mille** (`1‰`, `1/1000`) | **Normalize the concept, not the string.** Index `1‰`, `1/1000`, and `satu per seribu` as equivalent — they refer to the same penalty rate in this document. |
| **Contract IDs / NIP / DPPA** | Never tokenize on `/`, `-`, or internal spaces. Index whole, plus a normalized lowercase form. |
| **Dates** | Indonesian month names (`23 DESEMBER 2022`) → ISO 8601 in metadata; keep surface form in text. |
| **Abbreviations** | Auto-harvest via `yang selanjutnya disingkat (\w+)` / `selanjutnya disebut "([^"]+)"`. Build a document-scoped glossary. Expand *additively* in the BM25 field. |
| **Org / person names** | Preserve casing and full form. Do not stem. |
| **Template placeholders** (`…`, `[...]`) | Tag `is_template_placeholder`. Exclude from RAG context by default, or the model will answer questions with `.........`. |
| **Legal citations** (`UU No. 2 Tahun 2017`) | Extract as typed entities — enables "which regulations does this contract invoke". |

---

## H. Search and RAG Strategy

### H.1 Retrieval architecture

**Hybrid, sparse-weighted, with RRF fusion.** Three reasons specific to this corpus:

1. The corpus is small (~400–600 chunks from one contract). BM25's IDF statistics are most distinctive at exactly this scale (C.3).
2. The content is jargon-dense — precisely the profile where sparse retrieval should get more candidates (C.3).
3. The dominant query pattern is *finding one clause inside a long document*, where lexical matching is hardest to beat (C.4).

```
query
 ├─► BM25   (text_bm25,  top_k=100)  ─┐
 ├─► Dense  (text_embed, top_k=30)   ─┼─► RRF (k=60, α_sparse=0.6 / α_dense=0.4)
 └─► Metadata pre-filter               ┘        │
     (sub_document, clause_path,               ▼
      deontic_modality, fiscal_year)      cross-encoder rerank (top 10)
                                                │
                                                ▼
                                     parent-expand → RAG context
```

Add a **query router**: if the query matches a clause-reference regex (`pasal 37`, `SSKK 6.3`), bypass retrieval entirely and do a direct metadata lookup on `clause_path`. This is the highest-frequency query type in contract search and deserves an exact-match path rather than a probabilistic one.

### H.2 Chunking strategy

**Clause-aware, not fixed-size.** Your document has machine-detectable semantic boundaries (validated: 79/79 clauses). Throwing that away for a 512-token sliding window would be a genuine loss.

| Rule | Detail |
|---|---|
| **Base unit** | One SSUK sub-clause (`37.2`) = one chunk. This is the natural legal unit of reference. |
| **Small clauses** | If a sub-clause is < 80 tokens, merge with siblings under the same parent, up to ~350 tokens. Prevents a corpus of unretrievable fragments. |
| **Long clauses** | If > 450 tokens, split at item boundaries (`a.`, `b.`, `1)`), never mid-sentence. Every part inherits full `clause_path` and carries `chunk_index`/`chunk_total`. |
| **Never split** | A table row. A signature block. A `label : value` pair. A definition (SSUK 1.x). |
| **Contextual header** | Every chunk is prefixed with its breadcrumb: `[Kontrak 08/... › SSUK › D. Perubahan Kontrak › 37.2]`. Cheap, and directly implements the document-level enrichment that measurably improves retrieval (C.3). |
| **Overlap** | ~0 for standalone clauses; instead use `prev_node_id`/`next_node_id` for **parent expansion at retrieval time**. Cleaner than duplicating text, and gives surrounding context on demand. |
| **Tables** | One chunk per logical row for SSKK (each row is a self-contained provision). Whole-table chunk for small annex tables, plus a table-summary chunk. |
| **Cross-links** | SSKK chunks carry `refs_ssuk`; retrieving an SSUK clause should also surface its SSKK override. **This is a legal correctness requirement, not an enhancement** — the specific conditions modify the general ones. |

### H.3 What gets embedded

Embed `text_embed` — lightly normalized, case-preserved, with the contextual header. **Do not embed the stemmed BM25 field.** Stemmed text is out-of-distribution for every pretrained embedder; passing `sedia dapat alih kerja utama` to BGE-M3 will produce a materially worse vector than the natural sentence.

**Model recommendation: `BAAI/bge-m3`** (8192-token context, 100+ languages, strong on Indonesian, permissive license). Benchmark against `LazarusNLP/all-indo-e5-small-v4` on your own queries — the Indonesian-tuned smaller model may match it at a fraction of the compute, and only your own eval set can settle that.

### H.4 Source of truth vs. derived — explicit

| Layer | Status | Rule |
|---|---|---|
| `Rancangan_Kontrak.pdf` + SHA-256 | **Legal original** | Archived, never modified |
| `raw_extraction.json` → `text_raw` | **Source of truth** | Immutable. Every citation resolves here. Displayed to users verbatim. |
| `text_display` | Derived | Regenerable |
| `text_bm25` | **Search artifact only** | Never displayed. Never sent to the LLM. |
| `text_embed` / vectors | **Search artifact only** | Never displayed |
| `chunks.jsonl` | Derived, versioned | Rebuild on any schema change |
| **RAG context sent to LLM** | **Must be `text_raw`** | Not the cleaned text — the model needs the real clause with its numbers, negations, and identifiers intact |

That last row is the one most often gotten wrong. Preprocessing exists to help the *retriever find* the chunk. Once found, the **original text** is what goes to the model.

---

## I. Validation Strategy

### I.1 Coverage and conservation checks

| Check | Method | Threshold |
|---|---|---|
| Character conservation | Σ chars in structure nodes ÷ Σ chars in `raw_text` | ≥ 99.5% |
| No page dropped | Every page 1–74 produces ≥1 node (except p74, whitelisted) | 100% |
| No orphan text | Text in `raw_text` not assigned to any node | ≤ 0.5% |
| No duplication | Same char span claimed by 2 nodes | 0 |
| Token-set equality | `set(tokens(raw))` vs `set(tokens(reassembled))` | Symmetric difference = 0 |

### I.2 Dual-parser cross-validation

Extract every page with **both** pdfplumber and `pdftotext -layout` (independent implementations). Normalize whitespace, then compute Levenshtein ratio per page.

- **≥ 0.98** → accept
- **0.90–0.98** → flag for human review
- **< 0.90** → hard fail, halt pipeline

Two independent engines agreeing on the same character sequence is strong evidence the text layer was read correctly.

### I.3 Structural invariants (the highest-value checks)

These are cheap, deterministic, and catch the failures that matter:

| Invariant | Expected | Rationale |
|---|---|---|
| **SSUK clause count** | Exactly **79** | Empirically validated |
| **SSUK sequence** | 1…79, no gaps, no duplicates | Validated — zero gaps found |
| **Sub-clause monotonicity** | Within clause N, sub-clauses N.1, N.2… strictly increasing | Detects column-split failure |
| **Surat Perjanjian** | Exactly 5 `Pasal` | Verified |
| **Sub-documents** | Exactly 5 | Verified |
| **Orphan sub-clauses** | Every `N.M` has parent `N` | Detects heading loss |
| **SSKK refs resolve** | Every `refs_ssuk` points to a real SSUK node | Detects cross-link corruption |
| **Page-label continuity** | Lampiran B labels 11, 12, 13 sequential | Detects footer misparse |

**The 79-clause invariant is the single most valuable test in the suite.** If the two-column splitter degrades — because of a font change, a geometry variation, or a threshold bug — the clause count will drift, and this check fails loudly instead of silently producing a corrupted corpus.

### I.4 Content-level checks

| Check | Method |
|---|---|
| Broken paragraphs | Node text ending mid-sentence (no terminal punctuation, next node not a sub-item) → flag |
| Reading order | `reading_order` monotonic w.r.t. (page, top, x0) within each column |
| Header/footer leakage | Letterhead string must appear 0 times in any `structure` node |
| Table integrity | Every row's cell count == header count; else flag |
| Encoding sanity | Assert `‰` count == 3, `…` count == 338, no U+FFFD (replacement char) anywhere |
| Identifier survival | Regex-assert `08/PUPRPRKP-B.PNK/SP-PPK` and `19810618 200604 2 008` survive intact into `text_display` |
| Negation preservation | **Assert `tidak` count in `text_bm25` ≥ 95% of count in `text_raw`** — direct regression guard against the G.1 defect |
| Placeholder detection | Node containing `…{3,}` or `[...]` must have `is_template_placeholder: true` |

### I.5 Retrieval quality

Build a **golden query set of 30–50 real Indonesian questions** with hand-labeled correct clauses. Must include:
- Direct clause lookups (`"apa isi pasal 37.2"`)
- Conceptual queries (`"bagaimana ketentuan denda keterlambatan"`)
- **Negation-sensitive queries** (`"apa yang tidak boleh disubkontrakkan"`) ← the G.1 regression test
- Numeric queries (`"berapa besaran denda"`)
- Cross-reference queries (`"ketentuan korespondensi"` — must return both SSUK 4.1 and its SSKK row)

Track **Recall@5, Recall@10, MRR, nDCG@10**. Measure BM25-only, dense-only, and hybrid separately so you can prove the hybrid is earning its complexity rather than assuming it.

### I.6 Human review

Sample **10% of nodes stratified by sub-document** (guaranteeing coverage of all five). For each: render the page region from `bbox` with PyMuPDF, display side-by-side with extracted text, and have a reviewer mark correct/incorrect. Target ≥ 99% node-level accuracy before production. Prioritize the SSUK two-column pages — that is where the risk is concentrated.

---

## J. Implementation Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0. Scaffold** *(0.5d)* | Repo, deps pinned, structured logging (JSON lines), config-as-code, golden-file test harness | `pytest` green on an empty pipeline |
| **1. Inspect & route** *(1d)* | `manifest.json`; per-page geometry, font, char-density; native-vs-OCR router | Correctly routes all 74 pages to `native`; correctly routes a synthetic scanned page to `ocr` |
| **2. Raw extraction** *(3–4d)* ⚠️ *highest risk* | Sub-document segmenter; **two-column splitter**; ruled-table extractor; `raw_extraction.json` | **79/79 SSUK clauses, zero gaps**; dual-parser ratio ≥0.98 on all pages; char conservation ≥99.5% |
| **3. Hierarchy & entities** *(2–3d)* | Full node tree with `clause_path`; page-break stitching; regex entity tagger; abbreviation glossary; SSKK↔SSUK cross-links | All structural invariants (I.3) pass; contract no. + NIP survive intact |
| **4. Validation harness** *(1–2d)* | Automated I.1–I.4 suite as a CI gate | Pipeline **halts** on any hard-fail; report artifact emitted |
| **5. Preprocessing** *(2d)* | Custom legal stoplist; entity-protection mask/restore; `text_display` / `text_bm25` / `text_embed` | **Negation-preservation test passes**; `Pasal 37.2` and `Rp. …,00` survive the BM25 path |
| **6. Chunking** *(1–2d)* | `chunks.jsonl` with parent links + contextual headers | No chunk splits a table row or sentence; token distribution within 80–450 |
| **7. Retrieval prototype** *(2d)* | In-memory `rank_bm25` + BGE-M3; RRF fusion; golden query set | Hybrid beats both single-mode baselines on Recall@10 |
| **8. Production index** *(2–3d)* | Postgres + pgvector + tsvector; atomic dual-index writes; query router | Reindex is idempotent; sparse/dense never diverge |
| **9. RAG integration** *(2d)* | Context assembly from `text_raw`; parent expansion; citation rendering (PDF page **and** printed label) | Every answer cites a resolvable `node_id` |
| **10. Human review & harden** *(2–3d)* | 10% stratified review; error taxonomy; runbook | ≥99% node accuracy; all defects triaged |

**Estimated: 18–25 working days.** Phase 2 is the critical path — budget generously; everything downstream depends on getting the two-column split right.

**Generalization checkpoint:** after Phase 4, run the pipeline against 2–3 *different* Indonesian government contracts. This template (Perpres 16/2018 standard form) is widely reused, so the structure should transfer — but you want to discover the exceptions before building the search layer on top.

---

## K. Risks and Trade-offs

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Negation stripped → inverted legal meaning** | **Was near-certain** with the default recipe | **Critical** | Custom stoplist; automated negation-preservation assertion in CI (I.4) |
| 2 | Two-column splitter degrades on layout variation | Medium | **Critical** | 79-clause invariant as a hard gate; fractional (not absolute) coordinate thresholds |
| 3 | Template placeholders answered as facts | High if unhandled | High | `is_template_placeholder` flag; excluded from RAG context; `document_status: draft_template` surfaced in every answer |
| 4 | Stemming shatters clause refs / currency | High | High | Entity-protection mask/restore; stemming confined to `text_bm25` |
| 5 | Hard-coded coordinates break on 610pt pages | Medium | High | All thresholds as fractions of per-page width; assert on geometry variance |
| 6 | Citations point to wrong page (dual numbering) | Medium | Medium-High | Store `page` **and** `page_label`; render both |
| 7 | SSKK override missed → general rule returned instead of specific | Medium | **High (legal)** | Materialized `refs_ssuk` cross-links; co-retrieval enforced |
| 8 | Sparse/dense index drift | Medium | Medium | Single Postgres transaction for both writes |
| 9 | Sastrawi dictionary is CC-BY-NC-SA (non-commercial) | Low | Medium-High | Legal review before commercialization; consider a Snowball-based or custom stemmer as a swap-in |
| 10 | Over-engineering for one document | Medium | Medium | Phase 7 uses in-memory BM25 — validate retrieval quality *before* provisioning infrastructure |
| 11 | PyMuPDF AGPL contamination | Low | Medium | Confine PyMuPDF to dev-time QA rasterization; never ship it in the runtime path |
| 12 | Silent regression on pipeline change | Medium | High | Golden-file tests: freeze `raw_extraction.json` for this document, diff on every commit |

### Explicit trade-offs accepted

- **pdfplumber over PyMuPDF**: giving up ~10× speed for MIT licensing and superior coordinate/table handling. At 74 pages, speed is a non-issue; licensing and correctness are not.
- **Clause-aware over fixed-size chunking**: more code and more edge cases, in exchange for chunks that align with how lawyers actually cite. Justified because the boundaries are already validated as machine-detectable.
- **Hybrid over dense-only**: two indexes to maintain, for a large recall gain on a corpus full of identifiers where dense retrieval is known to be weak.
- **Postgres over Elasticsearch**: weaker BM25 tuning, in exchange for transactional consistency and one system instead of two. Revisit if the corpus grows past a few thousand contracts.
- **Three JSON files over one**: more artifacts to manage, in exchange for a genuinely immutable source of truth and regenerable derivations.

---

## L. Sources and References

**PDF parsing and extraction**
1. *A Comparative Study of PDF Parsing Tools Across Diverse Document Categories* — arXiv:2410.09871 — https://arxiv.org/html/2410.09871v1 (10 parsers, DocLayNet, 6 categories incl. Tender)
2. pdfplumber — PyPI / GitHub — https://pypi.org/project/pdfplumber/
3. PyMuPDF documentation, Features Comparison — https://pymupdf.readthedocs.io/en/latest/about.html
4. *PyMuPDF vs pdfplumber (2026): the speed-vs-license tradeoff, benchmarked* — https://pdfmux.com/blog/pymupdf-vs-pdfplumber/
5. *pdfplumber vs PyMuPDF: Which Python Library Produces Cleaner Markdown?* — https://www.file2markdown.ai/blog/pdfplumber-vs-pymupdf
6. *Python PDF library comparison (2026)* — https://www.nutrient.io/blog/best-python-pdf-libraries/
7. *Technical Comparison — Python Libraries for Document Parsing* — https://medium.com/@hchenna/technical-comparison-python-libraries-for-document-parsing-318d2c89c44e

**Indonesian NLP**
8. Sastrawi (Python) — PyPI — https://pypi.org/project/Sastrawi/
9. Sastrawi (upstream PHP, inactive) — https://github.com/sastrawi/sastrawi
10. Asian, J. (2007). *Effective Techniques for Indonesian Text Retrieval*. PhD thesis, RMIT University — foundational for the Nazief-Andriani/confix-stripping algorithm Sastrawi implements
11. Arifin, A.Z., Mahendra, I.P.A.K., Ciptaningtyas, H.T. (2009). *Enhanced Confix Stripping Stemmer and Ants Algorithm for Classifying News Document in Indonesian Language*, ICTS
12. Tahitoe, A.D., Purwitasari, D. (2010). *Implementasi Modifikasi Enhanced Confix Stripping Stemmer Untuk Bahasa Indonesia dengan Metode Corpus Based Stemming*, ITS Surabaya
13. *The Effect of Stemming and Removal of Stopwords on the Accuracy of Sentiment Analysis on Indonesian-language Texts* — https://www.researchgate.net/publication/337321725
14. *Algoritma Okapi BM25 dalam Sistem Pencarian Informasi Berbasis Teks* — Insand Comtech — https://ejournal.unira.ac.id/index.php/insand_comtech/article/view/2649
15. LazarusNLP Indonesian Sentence Embeddings — https://huggingface.co/collections/LazarusNLP/indonesian-sentence-embedding-6541fce662e82d932ff360c5

**Retrieval, hybrid search, and RAG**
16. *From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents* — arXiv:2604.01733 — https://arxiv.org/html/2604.01733v1
17. *Towards Reliable Retrieval in RAG Systems for Large Legal Datasets* — arXiv:2510.06999 — https://arxiv.org/pdf/2510.06999
18. *Hybrid Search for RAG: Combining BM25 and Dense Vector Search (2026 Guide)* — https://denser.ai/blog/hybrid-search-for-rag/
19. *Hybrid Search Guide: Vectors & Full-Text* — Supermemory — https://supermemory.ai/blog/hybrid-search-guide/
20. *Hybrid Search in RAG: Dense + Sparse (BM25/SPLADE), RRF, and When to Use Which* — https://blog.gopenai.com/hybrid-search-in-rag-dense-sparse-bm25-splade-reciprocal-rank-fusion-and-when-to-use-which-fafe4fd6156e
21. *Hybrid Search for RAG: Vector + Keyword + Reranking Guide 2026* — https://www.buildmvpfast.com/blog/hybrid-search-rag-vector-keyword-reranking-2026

**Embeddings**
22. BGE-M3 / FlagEmbedding — BAAI — https://huggingface.co/BAAI/bge-m3
23. *8 Embedding Models Compared for Production RAG [2026 Benchmark]* — https://tensoria.fr/en/blog/embedding-models-2026-guide
24. *The Best Open-Source Embedding Models in 2026* — BentoML — https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
25. *Bekko Embedding: how small can a multilingual retrieval model be?* — https://huggingface.co/blog/hotchpotch/bekko-embedding

**Primary source**
26. `Rancangan_Kontrak.pdf` — 74pp, Kontrak Harga Satuan no. 08/PUPRPRKP-B.PNK/SP-PPK, Dinas PUPRPRKP Kabupaten Sukamara, TA 2023. All structural findings, character inventories, clause counts, and Sastrawi behavioral tests in sections A, B, and G were measured directly against this file.

---

### Quick reference — the three things that matter most

1. **No OCR.** Native text layer is complete and perfect on all 74 pages. Adding OCR would degrade accuracy.
2. **`tidak` is in Sastrawi's default stoplist.** Using it as-is will invert the meaning of prohibition clauses. Build a custom legal stoplist before writing any preprocessing code.
3. **75% of the document is a borderless two-column table.** Naive extraction scrambles it; coordinate-based splitting recovers it perfectly (79/79 clauses validated). This is where to spend your engineering time.
