from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "output" / "raw"
PDF_OUT = PROJECT_DIR / "pdfs" / "synthetic"
TRUTH_OUT = PROJECT_DIR / "ground_truth" / "synthetic"

PAGE_W, PAGE_H = 595.0, 842.0
MARGIN = 72.0
LINE_H = 14.0
FONT, FONT_SIZE = "helv", 10.0
PROSE_SEED = 20260921


# --------------------------------------------------------------------------
# the authoring tree
# --------------------------------------------------------------------------

@dataclass
class Unit:

    marker: str
    label: str
    title: str = ""
    body: str = ""
    children: list["Unit"] = field(default_factory=list)
    caps_body: bool = False
    body_en: str = ""

    def walk(self, ancestors: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], "Unit"]]:
        path = ancestors + (self.label,) if self.label else ancestors
        out = [(path, self)]
        for child in self.children:
            out.extend(child.walk(path))
        return out


@dataclass
class Specimen:
    name: str
    description: str
    traps: list[str]
    units: list[Unit]
    preamble: list[str] = field(default_factory=list)
    two_column: bool = False
    parallel_columns: bool = False
    table_of_contents: bool = False
    image_only: bool = False

    def expected_units(self) -> list[dict]:
        out = []
        for unit in self.units:
            for path, node in unit.walk():
                out.append({
                    "path": list(path),
                    "parent_path": list(path[:-1]),
                    "marker": node.marker,
                    "title": node.title,
                    "body": node.body,
                })
        return out


# --------------------------------------------------------------------------
# prose lifted from the real specimens
# --------------------------------------------------------------------------

def _prose_pool() -> list[str]:
    """Sentences from the real corpus, deduped and ordered deterministically."""
    sentences: set[str] = set()
    for path in sorted(RAW_DIR.glob("*_raw.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for node in document.get("structure") or []:
            text = re.sub(r"\s+", " ", node.get("text_raw") or "").strip()
            for sentence in re.split(r"(?<=[.;])\s+", text):
                sentence = sentence.strip()
                if 60 <= len(sentence) <= 220 and sentence[0].isupper():
                    sentences.add(sentence)
    return sorted(sentences)


class Prose:
    def __init__(self) -> None:
        self.pool = _prose_pool()
        self.rng = random.Random(PROSE_SEED)
        if not self.pool:
            raise SystemExit("no prose available: run the pipeline over pdfs/ first (output/raw must exist)")

    def paragraph(self, sentences: int = 2) -> str:
        return " ".join(self.rng.choice(self.pool) for _ in range(sentences))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _wrap(text: str, width: float) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if fitz.get_text_length(candidate, fontname=FONT, fontsize=FONT_SIZE) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class Writer:

    def __init__(self, doc: fitz.Document, two_column: bool) -> None:
        self.doc = doc
        self.two_column = two_column
        self.label_x = MARGIN
        self.body_x = MARGIN + 110.0 if two_column else MARGIN
        self.body_w = PAGE_W - MARGIN - self.body_x
        self.page = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN

    def _room(self, lines: int) -> None:
        if self.y + lines * LINE_H > PAGE_H - MARGIN:
            self.new_page()

    def line(self, text: str, x: float | None = None, gap_after: float = 0.0) -> None:
        self._room(1)
        self.page.insert_text((x if x is not None else self.body_x, self.y), text,
                              fontname=FONT, fontsize=FONT_SIZE)
        self.y += LINE_H + gap_after

    def block(self, text: str, x: float, width: float, gap_after: float = 4.0) -> None:
        for line in _wrap(text, width):
            self.line(line, x=x)
        self.y += gap_after

    def heading_and_body(self, marker: str, title: str, body: str) -> None:
        if self.two_column and title:
            self._room(max(2, len(_wrap(body, self.body_w)) if body else 1))
            start_y = self.y
            for i, line in enumerate(_wrap(f"{marker} {title}", 100.0)):
                self.page.insert_text((self.label_x, start_y + i * LINE_H), line,
                                      fontname=FONT, fontsize=FONT_SIZE)
            if body:
                self.block(body, self.body_x, self.body_w)
            else:
                self.y = start_y + LINE_H
        else:
            head = f"{marker} {title}".strip() if title else marker
            self.block(head if not body else f"{head}", self.label_x, PAGE_W - 2 * MARGIN, gap_after=0.0)
            if body:
                self.block(body, self.label_x, PAGE_W - 2 * MARGIN)


def _render_units(writer: Writer, units: list[Unit], depth: int = 0) -> None:
    for unit in units:
        body = unit.body.upper() if unit.caps_body else unit.body
        indent = writer.label_x + (0 if writer.two_column else depth * 16.0)
        if writer.two_column:
            writer.heading_and_body(unit.marker, unit.title, body)
        elif unit.title:
            writer.block(f"{unit.marker} {unit.title}", indent, PAGE_W - MARGIN - indent, gap_after=0.0)
            if body:
                writer.block(body, indent + 12.0, PAGE_W - MARGIN - indent - 12.0)
        else:
            writer.block(f"{unit.marker} {body}".strip(), indent, PAGE_W - MARGIN - indent)
        _render_units(writer, unit.children, depth + 1)


def _render_parallel(writer: Writer, units: list[Unit]) -> None:
    gap = 24.0
    column_w = (PAGE_W - 2 * MARGIN - gap) / 2
    right_x = MARGIN + column_w + gap
    for unit in units:
        writer.block(f"{unit.marker} {unit.title}".strip(), MARGIN, PAGE_W - 2 * MARGIN, gap_after=2.0)
        left_lines = _wrap(unit.body, column_w)
        right_lines = _wrap(unit.body_en or unit.body, column_w)
        writer._room(max(len(left_lines), len(right_lines)) + 1)
        start_y = writer.y
        for i, line in enumerate(left_lines):
            writer.page.insert_text((MARGIN, start_y + i * LINE_H), line, fontname=FONT, fontsize=FONT_SIZE)
        for i, line in enumerate(right_lines):
            writer.page.insert_text((right_x, start_y + i * LINE_H), line, fontname=FONT, fontsize=FONT_SIZE)
        writer.y = start_y + max(len(left_lines), len(right_lines)) * LINE_H + 8.0
        _render_parallel(writer, unit.children)


def _render_toc(writer: Writer, specimen: Specimen) -> None:
    writer.line("DAFTAR ISI", x=MARGIN, gap_after=8.0)
    page_no = 2
    for unit in specimen.units:
        writer.line(f"{unit.marker} {unit.title} .................. {page_no}", x=MARGIN)
        page_no += 1
    writer.new_page()


def render(specimen: Specimen) -> fitz.Document:
    doc = fitz.open()
    writer = Writer(doc, specimen.two_column)
    if specimen.table_of_contents:
        _render_toc(writer, specimen)
    for line in specimen.preamble:
        writer.block(line, MARGIN, PAGE_W - 2 * MARGIN)
    if specimen.preamble:
        writer.y += 6.0
    if specimen.parallel_columns:
        _render_parallel(writer, specimen.units)
    else:
        _render_units(writer, specimen.units)

    if specimen.image_only:
        scanned = fitz.open()
        for page in doc:
            pixmap = page.get_pixmap(dpi=200)
            new_page = scanned.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(page.rect, pixmap=pixmap)
        doc.close()
        return scanned
    return doc


# --------------------------------------------------------------------------
# the specimens
# --------------------------------------------------------------------------

def build_specimens() -> list[Specimen]:
    prose = Prose()
    specimens: list[Specimen] = []

    specimens.append(Specimen(
        name="inverted_nesting",
        description="Perjanjian Kerja Sama with I./II. parts, A. sections, 1. clauses, a. and 1) below — "
                    "letters ABOVE digits, the reverse of the Perpres nesting order.",
        traps=["numbering depth order", "roman part headings", "no SSUK/SSKK markers"],
        preamble=["PERJANJIAN KERJA SAMA", "Nomor : 45/PKS/DIR/2024", prose.paragraph(2)],
        units=[
            Unit("I.", "I", "KETENTUAN UMUM", children=[
                Unit("A.", "A", "Definisi dan Ruang Lingkup", children=[
                    Unit("1.", "1", "Definisi", prose.paragraph(2), children=[
                        Unit("a.", "a", "", prose.paragraph(1)),
                        Unit("b.", "b", "", prose.paragraph(1), children=[
                            Unit("1)", "1", "", prose.paragraph(1)),
                            Unit("2)", "2", "", prose.paragraph(1)),
                        ]),
                    ]),
                    Unit("2.", "2", "Ruang Lingkup", prose.paragraph(2)),
                ]),
                Unit("B.", "B", "Hak dan Kewajiban", children=[
                    Unit("1.", "1", "Hak Pihak Pertama", prose.paragraph(2)),
                    Unit("2.", "2", "Kewajiban Pihak Kedua", prose.paragraph(2)),
                ]),
            ]),
            Unit("II.", "II", "PELAKSANAAN", children=[
                Unit("A.", "A", "Jangka Waktu", children=[
                    Unit("1.", "1", "Masa Berlaku", "Perjanjian ini berlaku selama 90 (sembilan puluh) hari kerja."),
                    Unit("2.", "2", "Perpanjangan", prose.paragraph(2)),
                ]),
            ]),
        ],
    ))

    specimens.append(Specimen(
        name="toc_and_caps_body",
        description="Contract opening with a DAFTAR ISI page that repeats every heading verbatim, and clauses "
                    "whose bodies are typed in ALL CAPS (older typewritten house style).",
        traps=["table of contents before the body", "ALL-CAPS body resets the tree stack",
               "caps party names mid-clause"],
        table_of_contents=True,
        preamble=["SURAT PERJANJIAN PENGADAAN BARANG", "Nomor : 77/SP-BARANG/2024"],
        units=[
            Unit("Pasal 1", "Pasal 1", "KETENTUAN UMUM", prose.paragraph(2), children=[
                Unit("(1)", "1", "", prose.paragraph(1)),
                Unit("(2)", "2", "", "PARA PIHAK SEPAKAT BAHWA SELURUH KETENTUAN DALAM PERJANJIAN INI "
                                     "MENGIKAT SECARA HUKUM.", caps_body=True),
            ]),
            Unit("Pasal 2", "Pasal 2", "RUANG LINGKUP PEKERJAAN", prose.paragraph(2), children=[
                Unit("(1)", "1", "", prose.paragraph(2)),
                Unit("(2)", "2", "", "PT SUMBER MAKMUR SENTOSA TBK BERTINDAK SELAKU PENYEDIA BARANG.",
                     caps_body=True),
            ]),
            Unit("Pasal 3", "Pasal 3", "HARGA DAN PEMBAYARAN", prose.paragraph(2), children=[
                Unit("(1)", "1", "", "Nilai Kontrak : Rp. 1.500.000.000,00 (satu miliar lima ratus juta rupiah)."),
                Unit("(2)", "2", "", prose.paragraph(1)),
            ]),
        ],
    ))

    specimens.append(Specimen(
        name="private_parties_no_nip",
        description="Agreement between two private companies. Party block uses PIHAK PERTAMA/KEDUA markers, "
                    "the representatives are company directors with no NIP, and every value is filled in.",
        traps=["parties without a NIP", "filled contract value with an 'Rp.' prefix",
               "duration in hari kerja", "no government vocabulary"],
        preamble=[
            "PERJANJIAN PENGADAAN JASA",
            "Nomor Kontrak : 12/PPJ/LEGAL/2024",
            "Nama Pekerjaan : Pengembangan Sistem Informasi Terpadu",
            "Nilai Kontrak : Rp. 2.750.000.000,00 (dua miliar tujuh ratus lima puluh juta rupiah)",
            "",
            "PIHAK PERTAMA",
            "Nama : Andi Wijaya",
            "Jabatan : Direktur Utama PT Nusantara Digital",
            "Berkedudukan di : Jl. Gatot Subroto No. 12, Jakarta Selatan",
            "",
            "PIHAK KEDUA",
            "Nama : Rina Kartika",
            "Jabatan : Direktur PT Mitra Solusi Informatika",
            "Berkedudukan di : Jl. Asia Afrika No. 8, Bandung",
        ],
        units=[
            Unit("Pasal 1", "Pasal 1", "MAKSUD DAN TUJUAN", prose.paragraph(2)),
            Unit("Pasal 2", "Pasal 2", "JANGKA WAKTU", children=[
                Unit("(1)", "1", "", "Masa Pelaksanaan ditetapkan selama 120 (seratus dua puluh) hari kerja."),
                Unit("(2)", "2", "", "Masa Pemeliharaan ditetapkan selama 6 (enam) bulan."),
            ]),
            Unit("Pasal 3", "Pasal 3", "SANKSI", children=[
                Unit("(1)", "1", "", "Denda keterlambatan sebesar 1‰ (satu permil) dari nilai kontrak per hari."),
                Unit("(2)", "2", "", prose.paragraph(1)),
            ]),
        ],
    ))

    specimens.append(Specimen(
        name="bilingual_two_column",
        description="Bilingual licence agreement: the Indonesian and English texts run as two parallel body "
                    "columns of equal width, with no label gutter. Reading order must keep each language "
                    "whole instead of interleaving them line by line.",
        traps=["two body columns, neither of them a label gutter",
               "English text must not be spliced into the Indonesian sentence"],
        parallel_columns=True,
        preamble=["PERJANJIAN LISENSI / LICENCE AGREEMENT", "Nomor / Number : 9/LIC/2024"],
        units=[
            Unit("Pasal 1", "Pasal 1", "RUANG LINGKUP", prose.paragraph(2),
                 body_en="The scope of this Agreement covers the licensing of the software and the related "
                         "maintenance services agreed by the parties in writing."),
            Unit("Pasal 2", "Pasal 2", "PEMBAYARAN", prose.paragraph(2),
                 body_en="Payment shall be made in accordance with the milestones set out in the annex to "
                         "this Agreement within thirty days of an undisputed invoice."),
            Unit("Pasal 3", "Pasal 3", "HUKUM YANG BERLAKU",
                 "Perjanjian ini tunduk pada dan ditafsirkan menurut hukum Republik Indonesia.",
                 body_en="This Agreement shall be governed by and construed in accordance with the laws of "
                         "the Republic of Indonesia."),
        ],
    ))

    caps_paragraph = ("SELURUH KETENTUAN DALAM PASAL INI MENGIKAT PARA PIHAK SEJAK TANGGAL "
                      "PENANDATANGANAN KONTRAK INI.")
    specimens.append(Specimen(
        name="parts_toc_and_caps",
        description="Perpres-form contract whose DAFTAR ISI page lists SYARAT-SYARAT UMUM KONTRAK and "
                    "SYARAT-SYARAT KHUSUS KONTRAK before the real headings, and whose clauses contain "
                    "unmarked ALL-CAPS paragraphs.",
        traps=["contents page starts a sub-document early", "ALL-CAPS paragraph inside a clause body",
               "content before the first part marker"],
        preamble=[
            "DAFTAR ISI",
            "SURAT PERJANJIAN .................. 2",
            "SYARAT-SYARAT UMUM KONTRAK .................. 3",
            "SYARAT-SYARAT KHUSUS KONTRAK .................. 4",
            "",
            "SURAT PERJANJIAN",
            "Nomor : 21/SP/PUPR/2024",
            "Pejabat Penandatangan Kontrak dan Penyedia sepakat sebagai berikut.",
        ],
        units=[
            Unit("Pasal 1", "Pasal 1", "KETENTUAN UMUM", prose.paragraph(1), children=[
                Unit("(1)", "1", "", prose.paragraph(1)),
                Unit("(2)", "2", "", f"{prose.paragraph(1)} {caps_paragraph}"),
                Unit("(3)", "3", "", prose.paragraph(1)),
            ]),
            Unit("SYARAT-SYARAT UMUM KONTRAK", "", "", "", children=[
                Unit("1.", "1", "Definisi", prose.paragraph(1)),
                Unit("2.", "2", "Penerapan", prose.paragraph(1)),
            ]),
            Unit("SYARAT-SYARAT KHUSUS KONTRAK", "", "", "", children=[
                Unit("1.", "1", "Korespondensi", prose.paragraph(1)),
            ]),
        ],
    ))

    specimens.append(Specimen(
        name="scanned_image_only",
        description="Specimen 3 re-rendered as page images with no text layer — a simulated scan, so the OCR "
                    "pipeline can be scored against known structure for the first time.",
        traps=["no text layer", "OCR word boxes", "OCR reading order"],
        image_only=True,
        preamble=[
            "PERJANJIAN PENGADAAN JASA",
            "Nomor Kontrak : 12/PPJ/LEGAL/2024",
            "Nilai Kontrak : Rp. 2.750.000.000,00 (dua miliar tujuh ratus lima puluh juta rupiah)",
        ],
        units=[
            Unit("Pasal 1", "Pasal 1", "MAKSUD DAN TUJUAN", prose.paragraph(1)),
            Unit("Pasal 2", "Pasal 2", "JANGKA WAKTU", children=[
                Unit("(1)", "1", "", "Masa Pelaksanaan ditetapkan selama 120 (seratus dua puluh) hari kerja."),
                Unit("(2)", "2", "", "Masa Pemeliharaan ditetapkan selama 6 (enam) bulan."),
            ]),
        ],
    ))
    return specimens


def write(specimen: Specimen) -> tuple[Path, Path]:
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    TRUTH_OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_OUT / f"{specimen.name}.pdf"
    doc = render(specimen)
    doc.save(str(pdf_path), garbage=4, deflate=True)
    page_count = doc.page_count
    doc.close()

    truth_path = TRUTH_OUT / f"{specimen.name}.authored.json"
    truth_path.write_text(json.dumps({
        "_purpose": "Authored structure for a synthetic specimen. Exact by construction: the same tree rendered "
                    "the PDF. Score with `python -m pipeline.synthetic_score`.",
        "_caveat": "Synthetic. Clean typography and a perfect text layer, so it tests parsing logic, not "
                   "robustness to real-world PDFs. Prose is lifted from the real corpus; structure is invented.",
        "name": specimen.name,
        "description": specimen.description,
        "traps": specimen.traps,
        "pdf": f"pdfs/synthetic/{specimen.name}.pdf",
        "page_count": page_count,
        "engine": "ocr" if specimen.image_only else "native",
        "two_column": specimen.two_column,
        "units": specimen.expected_units(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pdf_path, truth_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic specimen set")
    parser.add_argument("--list", action="store_true", help="Describe the specimens without writing anything")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()

    specimens = [s for s in build_specimens() if not args.only or s.name in args.only]
    for specimen in specimens:
        if args.list:
            print(f"{specimen.name:26} {specimen.description}")
            print(f"{'':26} traps: {', '.join(specimen.traps)}")
            continue
        pdf_path, truth_path = write(specimen)
        print(f"wrote {pdf_path.relative_to(PROJECT_DIR)} and {truth_path.relative_to(PROJECT_DIR)} "
              f"({len(specimen.expected_units())} authored units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
