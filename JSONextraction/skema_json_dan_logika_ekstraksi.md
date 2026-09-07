# JSON Output Design & Generalized Extraction Logic
### For arbitrary Indonesian contracts — not just `Rancangan_Kontrak.pdf`

---

## 0. The core design shift

The previous analysis was written *against one document*. That produced findings like "expect exactly 79 SSUK clauses" — true for that file, useless for the next one.

The fix is to split the system into three layers with different stability guarantees:

| Layer | Stability | Varies per contract? |
|---|---|---|
| **Core envelope** — the 6 fields RAG always needs | **Fixed forever.** Same keys, same types, always present | No |
| **Structure tree** — generic recursive nodes | **Fixed shape**, variable depth and labels | Depth/labels vary |
| **Profile** — how to *find* the structure in this layout | Pluggable, swappable | Yes, entirely |

Only the third layer knows what "SSUK" means. The first two are contract-agnostic.

The practical rule that follows: **the schema never has a key named `ssuk_clause_number`.** It has `node_type: "clause"` and `label: "37.2"`. A profile decides that a bold left-column `37.` is a clause. A different profile decides that a centered `PASAL 12` is a clause. The consumer of the JSON never has to care.

### The guarantee that matters most

You named six things RAG always needs. Those get a **hard contract**:

> `core` is always present. Every field in it always exists as a key. A field is either a populated object with provenance, or `null` with a documented reason. It is **never missing, never a bare string, never silently empty**.

Downstream code can then be written as `doc.core.contract_number.value` without defensive checks, and a `null` is a real, auditable signal ("not present in this document") rather than a parse failure hiding as an absence.

---

## 1. Top-level output shape

Three files, as before, but now with the core envelope hoisted to the top of the raw layer.

```
raw_extraction.json   ← source of truth (immutable)
  ├── schema_version
  ├── source            (file identity, hashes, PDF metadata)
  ├── profile           (which extraction profile matched, and why)
  ├── core              ★ THE SIX GUARANTEED FIELDS
  ├── pages[]           (verbatim per-page text + geometry)
  ├── structure[]       (generic recursive node tree)
  ├── tables[]
  ├── entities[]        (all typed entities, document-wide)
  └── quality           (validation results, flags)

enriched.json         ← derived text views (regenerable)
chunks.jsonl          ← retrieval units (regenerable, versioned)
```

---

## 2. The `core` block — guaranteed fields

Every field follows the same **value object** pattern. This is the single most important convention in the schema:

```jsonc
{
  "value":       <normalized, typed value>,   // null if not found
  "raw":         "<verbatim surface string>", // exactly as printed
  "confidence":  0.0-1.0,
  "method":      "regex_labeled | regex_pattern | positional | table_cell | llm_fallback | manual",
  "evidence": {
    "node_id":   "n_0042",
    "page":      1,
    "page_label": null,
    "char_span": [120, 152],
    "bbox":      { "x0": 216.0, "top": 180.2, "x1": 420.5, "bottom": 194.0 },
    "context":   "...Nomor : 08/PUPRPRKP-B.PNK/SP-PPK Tanggal..."
  },
  "candidates": [ /* other values considered, with scores */ ],
  "flags":      ["template_placeholder"]
}
```

Why every field carries `raw` **and** `value`: `value` is what code queries (`2023-12-23`), `raw` is what a human verifies against the PDF (`23 DESEMBER 2022`). Why `evidence`: a contract system that can't show *where* it got a number is not usable for legal work. Why `candidates`: when extraction is ambiguous, silently picking one and discarding the rest destroys the information a reviewer needs.

### 2.1 Full `core` block

```jsonc
"core": {

  // ── 1. DOCUMENT TYPE ────────────────────────────────
  "document_type": {
    "value": "kontrak_konstruksi",        // controlled vocabulary
    "label_id": "Surat Perjanjian (Kontrak) Kerja Konstruksi",
    "subtype": "kontrak_harga_satuan",
    "raw": "SURAT PERJANJIAN\n(KONTRAK)",
    "confidence": 0.97,
    "method": "regex_labeled",
    "evidence": { "page": 1, "node_id": "n_0003", "char_span": [201, 228] },
    "candidates": [],
    "flags": []
  },

  // ── 2. CONTRACT NAME / SUBJECT ──────────────────────
  "contract_name": {
    "value": "Peningkatan Jalan Mekar Desa Natai Sedawak",
    "raw": "PEKERJAAN : PENINGKATAN JALAN MEKAR DESA NATAI SEDAWAK",
    "confidence": 0.94,
    "method": "regex_labeled",
    "label_matched": "PEKERJAAN",
    "evidence": { "page": 1, "node_id": "n_0009", "char_span": [612, 665] },
    "candidates": [
      { "value": "Pembangunan Jalan", "label_matched": "SUB KEGIATAN", "score": 0.61 }
    ],
    "flags": []
  },

  // ── 3. CONTRACT NUMBER ──────────────────────────────
  "contract_number": {
    "value": "08/PUPRPRKP-B.PNK/SP-PPK",
    "normalized": "08/PUPRPRKP-B.PNK/SP-PPK",
    "raw": "Nomor : 08/PUPRPRKP-B.PNK/SP-PPK",
    "confidence": 0.99,
    "method": "regex_labeled",
    "occurrence_count": 3,                 // ← corroboration signal
    "evidence": { "page": 1, "node_id": "n_0004", "char_span": [244, 268] },
    "candidates": [],
    "flags": []
  },

  // ── 4. PARTIES (organizations & signatories) ────────
  "parties": {
    "value": [
      {
        "party_id": "party_1",
        "role": "pihak_pertama",
        "role_label": "PPKom",
        "role_raw": "selanjutnya disebut \"PPKom\"",
        "organization": {
          "value": "Dinas Pekerjaan Umum dan Penataan Ruang dan Perumahan Rakyat dan Kawasan Permukiman Kabupaten Sukamara",
          "type": "government_agency",
          "confidence": 0.95
        },
        "representative": {
          "name": "GIAJENG WULANDARI, ST",
          "position": "Pejabat Pembuat Komitmen Bidang Bina Marga",
          "identifier": { "type": "NIP", "value": "19810618 200604 2 008" },
          "confidence": 0.96
        },
        "address": "Jl. Tjilik Riwut Km. 7,5 Sukamara",
        "authority_basis": {
          "document": "Surat Keputusan Kepala Dinas",
          "number": "945/140/2023",
          "date": "2023-04-03"
        },
        "evidence": { "page": 2, "node_id": "n_0021" }
      },
      {
        "party_id": "party_2",
        "role": "pihak_kedua",
        "role_label": "Penyedia",
        "organization": { "value": null, "type": "company", "confidence": 0.0 },
        "representative": { "name": null, "position": null, "identifier": null },
        "flags": ["template_placeholder"],
        "evidence": { "page": 2, "node_id": "n_0024" }
      }
    ],
    "confidence": 0.72,                    // ← lowered: party_2 unresolved
    "method": "regex_labeled",
    "flags": ["incomplete_parties"]
  },

  // ── 5. KEY DATES ────────────────────────────────────
  "key_dates": {
    "value": [
      { "type": "contract_date",     "date": null, "raw": "Tanggal : (Tgl Bulan 2023)",
        "confidence": 0.0, "flags": ["template_placeholder"] },
      { "type": "budget_doc_date",   "date": "2022-12-23", "raw": "23 DESEMBER 2022",
        "precision": "day", "confidence": 0.96, "label_matched": "TANGGAL",
        "related_to": "DPPA-SKPD 1.03.10.2.01.05" },
      { "type": "authority_date",    "date": "2023-04-03", "raw": "03 April 2023",
        "precision": "day", "confidence": 0.93 },
      { "type": "effective_date",    "date": null, "raw": "Tanggal (Awal 2023)",
        "confidence": 0.0, "flags": ["template_placeholder"] },
      { "type": "completion_date",   "date": null, "raw": "Tanggal (Akhir 2023)",
        "confidence": 0.0, "flags": ["template_placeholder"] },
      { "type": "fiscal_year",       "date": "2023", "precision": "year",
        "raw": "TAHUN ANGGARAN 2023", "confidence": 0.98 }
    ],
    "confidence": 0.55,
    "flags": ["multiple_dates_unresolved"]
  },

  // ── 6. KEY NUMBERS ──────────────────────────────────
  "key_numbers": {
    "value": [
      { "type": "contract_value", "amount": null, "currency": "IDR",
        "raw": "Rp. ..................,00", "confidence": 0.0,
        "flags": ["template_placeholder"] },
      { "type": "duration", "amount": 120, "unit": "hari_kalender",
        "raw": "120 (Seratus Dua Puluh) hari kalender", "confidence": 0.95,
        "words_check": "passed" },
      { "type": "penalty_rate", "amount": 0.001, "unit": "ratio",
        "raw": "1/1000 (satu per seribu)", "basis": "Nilai Kontrak sebelum PPN",
        "equivalent_forms": ["1‰", "0.1%", "satu per seribu"], "confidence": 0.97 },
      { "type": "reference_number", "subtype": "DPPA-SKPD",
        "value": "1.03.10.2.01.05", "raw": "1.03.10.2.01.05", "confidence": 0.98 }
    ],
    "confidence": 0.68,
    "flags": ["contract_value_missing"]
  },

  // ── extraction-wide status ──────────────────────────
  "_status": {
    "document_status": "draft_template",
    "fields_populated": 4,
    "fields_null": 2,
    "overall_confidence": 0.71,
    "requires_human_review": true,
    "review_reasons": ["contract_value_missing", "counterparty_unresolved"]
  }
}
```

### 2.2 Notes on specific fields

**`document_type` uses a controlled vocabulary.** Free-text here makes filtering impossible. Suggested starting set, extend as you onboard types:

```
kontrak_konstruksi · kontrak_pengadaan_barang · kontrak_jasa_konsultansi
kontrak_jasa_lainnya · surat_perintah_kerja · surat_pesanan
adendum · nota_kesepahaman · perjanjian_kerja_sama · surat_penunjukan
berita_acara · dokumen_lelang · unknown
```

`unknown` is a legitimate value. Forcing a guess is worse than admitting one.

**`parties` is an array, not `company_name`.** You said "company name", but Indonesian contracts have at minimum two sides, and often more (KSO/consortium members, guarantors, witnesses). This document's Lampiran A even has a subcontractor table. A flat `company_name` field would silently drop half the contract. The array with `role` handles two-party, KSO, and multi-party cases with one shape.

**`key_dates` and `key_numbers` are typed arrays, not fixed fields.** Which dates matter varies enormously — a construction contract cares about `effective_date`/`completion_date`; a lease cares about `commencement`/`expiry`/`rent_review`; an adendum cares about `original_contract_date`. A typed array with a controlled `type` vocabulary generalizes; a fixed `start_date`/`end_date` pair does not.

**`words_check` on numbers is a cheap accuracy win.** Indonesian contracts almost always write critical numbers twice: `120 (Seratus Dua Puluh)`, `Rp 500.000.000,00 (Lima Ratus Juta Rupiah)`. Parse both, compare, and you get free verification. Mismatch → flag for review. This catches OCR digit errors and extraction slips that nothing else would.

**`equivalent_forms` on rates.** This document expresses the same penalty as `1/1000`, `1‰`, and `satu per seribu` in different places. Normalizing to a single numeric `amount` plus a list of surface forms means a search for any variant finds the provision.

---

## 3. The generic structure tree

The tree is **recursive and label-agnostic**. No SSUK/SSKK vocabulary anywhere.

```jsonc
"structure": [
  {
    "node_id": "n_0512",                  // stable, sequential, opaque
    "parent_id": "n_0509",
    "depth": 3,

    "node_type": "clause",
    // generic vocabulary:
    //   part | chapter | section | article | clause | subclause
    //   list_item | paragraph | table | table_row | definition
    //   heading | preamble | recital | signature_block
    //   header | footer | page_number | annex | placeholder | unknown

    "label": "37.2",                       // as printed
    "label_normalized": "37.2",
    "numbering_style": "decimal_dotted",   // decimal_dotted | roman_upper |
                                           // latin_lower | paren_digit | bullet | none
    "title": null,                         // heading text if the node has one
    "path": ["Bagian D", "37", "37.2"],    // labels only — no domain words
    "path_display": "Bagian D. Perubahan Kontrak › 37. Perubahan Kontrak › 37.2",

    "text_raw": "Dalam hal tidak terjadi perubahan kondisi lapangan ...",

    "pages": [30],
    "page_labels": [],
    "spans_page_break": false,
    "bbox": { "page": 30, "x0": 226.4, "top": 88.1, "x1": 540.2, "bottom": 246.7 },
    "char_span": { "page": 30, "start": 0, "end": 412 },
    "reading_order": 512,

    "extraction": {
      "method": "native",                  // native | ocr | table_cell
      "confidence": 1.0,
      "detector": "two_column_left_label", // which profile rule fired
      "flags": []
    },

    "refs_out": [                           // cross-references found in text
      { "raw": "pasal 37.1", "resolved_node_id": "n_0510", "type": "internal" },
      { "raw": "UU No. 2 Tahun 2017", "resolved_node_id": null, "type": "external_law" }
    ],
    "children": ["n_0513", "n_0514", "n_0515"]
  }
]
```

**`detector` is the debugging field.** When a document extracts badly, you need to know *which rule* produced each node. Grouping bad nodes by `detector` tells you immediately whether the problem is one broken rule or a wrong profile.

**`refs_out` generalizes the SSKK↔SSUK cross-link.** Rather than a special-cased field, any node can reference any other. Resolution is a post-pass. External references (laws, regulations) resolve to `null` but stay recorded.

---

## 4. Extraction code logic — generalized

### 4.1 Pipeline stages

```
┌─ STAGE 1: PROBE ────────────────────────────────────────┐
│ Per page: font presence, char density, image coverage,   │
│ geometry, ruling-line count, word x0 histogram           │
│ → page_profile[] (no text decisions yet)                 │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 2: ROUTE (per page, never per document) ─────────┐
│  chars > 50 AND fonts present  → native                  │
│  chars ≈ 0 AND image ≥ 80%     → ocr                     │
│  otherwise                      → hybrid (both, merge)   │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 3: PROFILE SELECTION ────────────────────────────┐
│ Score each registered profile against document features. │
│ Best score above threshold wins; else generic fallback.  │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 4: LAYOUT SEGMENTATION (per page) ───────────────┐
│ Classify page layout: single_column | two_column |       │
│ ruled_table | form | mixed | blank                       │
│ Detected from x0 histogram + ruling lines, NOT hardcoded │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 5: BLOCK EXTRACTION ─────────────────────────────┐
│ Layout-appropriate extractor emits ordered text blocks   │
│ with bbox + font attributes. Still flat, no hierarchy.   │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 6: NUMBERING DETECTION & TREE BUILD ─────────────┐
│ Recognize numbering tokens → infer depth from            │
│ (style, indent, font weight) → build recursive tree      │
│ → stitch nodes across page breaks                        │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 7: ENTITY EXTRACTION ────────────────────────────┐
│ Cascade of strategies over the tree (§4.4)               │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 8: CORE FIELD RESOLUTION ────────────────────────┐
│ Promote entities → the six guaranteed core fields,       │
│ resolving conflicts by score (§4.5)                      │
└──────────────────────────────────────────────────────────┘
                          ▼
┌─ STAGE 9: VALIDATE → raw_extraction.json ───────────────┐
└──────────────────────────────────────────────────────────┘
```

### 4.2 Stage 4 — layout detection without hardcoding

The key generalization. Don't assume "pages 7–61 are two-column". **Detect it.**

```
For each page:
  1. Collect word x0 values, normalized to fraction of page width.
  2. Build a histogram (50 bins).
  3. Find modes (peaks above noise floor).

  IF ruling_lines > 20 AND form a grid
      → ruled_table
  ELIF two dominant modes, well-separated (gap > 0.15 of width),
       AND both sustain > 25% of lines on the page
      → two_column;  column boundary = midpoint of the gap
  ELIF ≥ 3 modes with short text runs between them
      → form / label-value layout
  ELIF one dominant mode
      → single_column
  ELIF char_count < 50
      → blank
  ELSE
      → mixed  (fall back to conservative single-column + flag)
```

Every threshold is a **fraction of page dimensions**, never an absolute point value. This is what protects against the 612pt/610pt/936pt variance found in the sample document — and against A4 vs Letter contracts generally.

**Column boundary is computed, not configured.** For the sample file this lands near `x0 ≈ 0.18 × width`; for a different contract it might be `0.30`. The code never knows the number in advance.

### 4.3 Stage 6 — numbering detection

A generic recognizer, ordered by specificity:

| Style | Pattern | Typical depth |
|---|---|---|
| `chapter_word` | `^(BAB|BAGIAN)\s+([IVXLC]+|\d+)` | 0 |
| `article_word` | `^PASAL\s+(\d+)` | 0–1 |
| `letter_upper` | `^([A-Z])\.\s+[A-Z]` | 0–1 |
| `decimal_dotted` | `^(\d+(?:\.\d+)+)\s` | depth = dot count |
| `decimal_plain` | `^(\d{1,3})\.\s` | 1 |
| `latin_lower` | `^([a-z])\.\s` | 2–3 |
| `paren_digit` | `^(\d+)\)\s` | 3–4 |
| `paren_latin` | `^([a-z])\)\s` | 3–4 |
| `roman_lower` | `^([ivxlc]+)\.\s` | 2–3 |
| `bullet` | `^[•▪◦-]\s` | leaf |

**Depth is not read off the table.** It's inferred from three signals combined:

1. **Intrinsic style rank** (a `decimal_dotted` with 2 dots is deeper than one with 1)
2. **Indentation** — `x0` relative to the parent node's `x0`
3. **Font weight/size** — bold and larger implies a heading

Then a **sequence validator** runs: within a parent, sibling labels must increase monotonically (`1, 2, 3` / `a, b, c`). A break in sequence means the depth inference was wrong — backtrack and retry with the next-best hypothesis. This self-correction is what makes the tree builder robust across layouts you haven't seen.

**Page-break stitching rule:** if the last node on page N has no terminal punctuation and the first block on page N+1 starts with a lowercase letter and carries no numbering token, merge them.

### 4.4 Stage 7 — entity extraction cascade

Each core field is attacked by strategies in order. **First strategy to clear its confidence threshold wins; the rest run anyway and populate `candidates`.**

```
STRATEGY 1 — Labeled lookup            (confidence 0.90–0.99)
  Indonesian contracts are heavily label:value formatted.
  Search for a label token, take the value after ':' or in the next column.
  Label dictionary per field, with synonyms.

STRATEGY 2 — Contextual pattern        (confidence 0.70–0.90)
  Regex over a window near a trigger phrase.
  e.g. contract number within 100 chars of "Nomor"/"No."

STRATEGY 3 — Positional heuristic      (confidence 0.50–0.75)
  Page-1 title block, largest font, centered → contract name.
  Signature block region → party names.

STRATEGY 4 — Structural                (confidence 0.60–0.85)
  Definition sections ("selanjutnya disebut X") → party role labels.
  Table cells keyed by header text.

STRATEGY 5 — LLM fallback              (confidence 0.40–0.80)
  Only when 1-4 all fail or disagree. Feed the first 2 pages +
  signature block, request strict JSON, REQUIRE the model to return
  a verbatim quote. Reject any value whose quote is not found in the
  source text — this blocks hallucination structurally.

STRATEGY 6 — Human review queue        (confidence 0.0)
  Field emitted as null with review_reason.
```

**Label dictionaries** (the generalization workhorse — extend, never replace):

```
contract_number:  Nomor · No. · Nomor Kontrak · Nomor Surat Perjanjian
                  Nomor SPK · Nomor Perjanjian · Nomor Dokumen
contract_name:    Pekerjaan · Nama Pekerjaan · Paket Pekerjaan · Kegiatan
                  Sub Kegiatan · Nama Paket · Perihal · Tentang · Objek Perjanjian
parties:          Nama · Jabatan · Berkedudukan di · Alamat · yang bertindak
                  untuk dan atas nama · selanjutnya disebut · PIHAK PERTAMA
                  PIHAK KEDUA · Penyedia · Pengguna Jasa · antara ... dengan
dates:            Tanggal · Tgl · pada tanggal · Tahun Anggaran · Masa Berlaku
                  Jangka Waktu · Mulai · Selesai · Berlaku sejak · s/d
values:           Nilai Perjanjian · Nilai Kontrak · Harga Kontrak · Jumlah
                  Nilai Pekerjaan · Total · Rp · sebesar · Pagu Anggaran
document_type:    (title-block match, first page, largest font)
```

**Number and date normalizers** must be Indonesian-aware:

- Currency: `.` = thousands, `,` = decimal. `Rp. 1.500.000.000,00` → `1500000000.00`. An English-convention parser produces `1.5` here — a nine-order-of-magnitude error, and a silent one.
- Months: `Januari…Desember`, plus abbreviations and all-caps forms.
- Number words: `Seratus Dua Puluh` → `120`, for the `words_check` cross-validation.
- Rates: unify `%`, `‰`, `x/y`, and `satu per seribu` to a decimal ratio.

### 4.5 Stage 8 — conflict resolution

When multiple strategies return different values for one field:

```
score = strategy_base_confidence
      × label_specificity        (exact label > synonym > inferred)
      × position_prior           (page 1 title block > body > annex)
      × corroboration_bonus      (× 1.15 if the same value appears ≥ 2 times)
      × format_validity          (does it match the expected shape?)
      × recency_penalty          (later occurrence in an adendum wins)

Highest score → core.<field>.value
All others    → core.<field>.candidates[]

IF top_score - second_score < 0.15
   → flags += "ambiguous"; requires_human_review = true
```

Keeping losing candidates is what makes review tractable — a reviewer picks from a short list instead of re-reading the PDF.

### 4.6 Profile registry

A profile bundles the layout hints for a document family:

```jsonc
{
  "profile_id": "perpres16_konstruksi_v1",
  "description": "Indonesian govt construction contract, Perpres 16/2018 standard form",
  "match_signals": [
    { "type": "text_contains", "value": "SYARAT-SYARAT UMUM KONTRAK", "weight": 0.4 },
    { "type": "text_contains", "value": "Pejabat Penandatangan Kontrak", "weight": 0.3 },
    { "type": "layout", "value": "two_column_dominant", "weight": 0.2 },
    { "type": "page_count_range", "value": [40, 200], "weight": 0.1 }
  ],
  "match_threshold": 0.6,
  "sub_document_markers": [
    { "name": "main_agreement", "start": "^SURAT PERJANJIAN" },
    { "name": "general_terms",  "start": "^SYARAT-SYARAT UMUM KONTRAK" },
    { "name": "special_terms",  "start": "^SYARAT-SYARAT KHUSUS KONTRAK" },
    { "name": "annex",          "start": "^LAMPIRAN\\s+[A-Z]" }
  ],
  "expected_invariants": {
    "clause_sequence_gapless": true,
    "min_clauses": 20
  }
}
```

**`generic_contract_v1` is the mandatory fallback profile.** When nothing matches above threshold, it runs single-column extraction + generic numbering detection + the full entity cascade. It will produce a shallower tree, but it **always populates `core`** — which is the part RAG depends on. Degraded structure is acceptable; a missing contract number is not.

Onboarding a new contract family becomes: write a profile JSON, add label synonyms, add an invariant. No changes to the schema, the tree builder, or anything downstream.

---

## 5. Chunking with the core attached

Every chunk carries a denormalized copy of `core` essentials in its metadata. This is deliberate duplication:

```jsonc
{
  "chunk_id": "kontrak-08::n_0512::0",
  "doc_id": "...",
  "text": "[Kontrak Konstruksi 08/PUPRPRKP-B.PNK/SP-PPK — Peningkatan Jalan Mekar Desa Natai Sedawak]\n[Bagian D › 37. Perubahan Kontrak › 37.2]\n\nDalam hal tidak terjadi perubahan...",
  "metadata": {
    "document_type": "kontrak_konstruksi",
    "contract_name": "Peningkatan Jalan Mekar Desa Natai Sedawak",
    "contract_number": "08/PUPRPRKP-B.PNK/SP-PPK",
    "party_names": ["Dinas PUPRPRKP Kabupaten Sukamara"],
    "fiscal_year": 2023,
    "effective_date": null,
    "contract_value": null,
    "node_id": "n_0512", "parent_node_id": "n_0509",
    "clause_path": "D/37/37.2", "label": "37.2", "depth": 3,
    "pages": [30], "page_labels": [],
    "document_status": "draft_template"
  }
}
```

Two reasons for the duplication. First, **metadata filtering**: "penalty clauses in 2023 construction contracts over Rp 1B" becomes a filter, not a semantic search. Second, **retrieval context**: a chunk retrieved in isolation is otherwise anonymous — the model receives clause text with no idea which contract it belongs to, which is how cross-contract answer contamination happens in multi-document RAG.

The contextual header embeds the contract identity into the vector itself, so a query naming the contract number gets lexical *and* semantic lift.

---

## 6. Validation, generalized

Document-specific invariants (like "exactly 79 clauses") move into the **profile**, not the code. The generic checks that apply to every contract:

| Check | Rule | Severity |
|---|---|---|
| Core presence | All 6 `core` keys exist | **hard fail** |
| Core typing | Each is a value object or explicit `null` | **hard fail** |
| Char conservation | Σ node chars ÷ Σ page chars ≥ 0.995 | hard fail |
| No text duplication | No char span claimed twice | hard fail |
| Tree integrity | No orphans; no cycles; `depth` consistent with `parent` | hard fail |
| Sibling sequence | Labels monotonic within each parent | warn → review |
| Reading order | Monotonic w.r.t. (page, column, top) | warn |
| Words-vs-digits | `120 (Seratus Dua Puluh)` agree | warn → review |
| Identifier survival | Contract number appears verbatim in `text_display` | hard fail |
| Negation preservation | `tidak`/`bukan`/`dilarang` count in BM25 field ≥ 95% of raw | **hard fail** |
| Encoding sanity | Zero U+FFFD replacement characters | hard fail |
| Placeholder tagging | Nodes with `…{3,}` or `[...]` flagged | warn |
| Profile invariants | Whatever the matched profile declares | warn → review |

**Confidence gate before RAG ingestion:** documents with `overall_confidence < 0.6` or `requires_human_review: true` are indexed but tagged `provisional`. Retrieval can then either exclude them or surface a warning — better than silently serving low-confidence extractions as fact.

---

## 7. What changed from the previous design

| Previous | Now | Why |
|---|---|---|
| `ssuk.c37.s2` node IDs | `n_0512` opaque IDs | Domain vocabulary in IDs breaks on non-SSUK contracts |
| `sub_document: "SSUK"` | `path[]` + profile-supplied names | Generic tree, profile-specific labels |
| `contract_metadata` buried in `document` | Top-level `core` with hard guarantees | RAG's required fields deserve a stable contract |
| Bare string values | Value objects with `raw` + `evidence` + `confidence` | Traceability and reviewability |
| `refs_ssuk` special field | Generic `refs_out[]` | Any node can reference any node |
| Hardcoded page ranges | Detected layout per page | Page ranges don't transfer between documents |
| Fixed `x0 < 110` threshold | Computed column boundary from x0 histogram | Absolute coordinates break on different page sizes |
| "79 clauses" in validation code | Invariant declared in the profile | Document-specific facts belong in config |
| `company_name` (implied singular) | `parties[]` array with roles | Contracts have ≥2 sides; KSO has more |
| Fixed date/number fields | Typed arrays with controlled vocabularies | Which dates matter varies by contract type |

---

## 8. Open decisions for you

1. **Controlled vocabularies** — the `document_type` and date/number `type` lists above are a starting proposal. They should be reviewed against the actual mix of contracts you'll ingest, since adding values later is easy but *changing* them means re-extraction.

2. **LLM fallback (Strategy 5)** — worth deciding early whether it's in scope. It substantially raises core-field recall on unusual layouts, but adds a dependency and a cost per document. The verbatim-quote validation makes it safe; the question is whether you want it in v1.

3. **Confidence thresholds** — the numbers above (0.6 review gate, 0.15 ambiguity margin) are reasonable defaults, not tuned values. They should be calibrated once you have 20–30 documents with human-reviewed ground truth.

4. **Party identity resolution** — whether "Dinas PUPRPRKP Kabupaten Sukamara" and "Dinas Pekerjaan Umum dan Penataan Ruang dan Perumahan Rakyat dan Kawasan Permukiman Kabupaten Sukamara" should resolve to one canonical entity across documents. Worth doing eventually; probably not v1.
