from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ADVERSARIAL = PROJECT_DIR / "pdfs" / "adversarial"
SYNTHETIC = PROJECT_DIR / "pdfs" / "synthetic"


def _run_cli(*args: str, module: str = "pipeline.main") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


class FailureRecordTest(unittest.TestCase):
    def _assert_failure_record(self, out_dir: Path, stem: str) -> dict:
        status_path = out_dir / f"{stem}_status.json"
        self.assertTrue(status_path.exists(), f"no failure record for {stem}")
        record = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(record["pipeline_status"], "failed")
        self.assertTrue(record["error_class"])
        self.assertEqual(record["source"]["file"], f"{stem}.pdf")
        self.assertFalse((out_dir / f"{stem}_raw.json").exists())
        return record

    def test_bad_inputs_each_write_a_failure_record(self):
        for name in ("truncated", "encrypted", "zero_byte"):
            with self.subTest(pdf=name), tempfile.TemporaryDirectory() as tmp:
                proc = _run_cli(str(ADVERSARIAL / f"{name}.pdf"), "--out", tmp)
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self._assert_failure_record(Path(tmp), name)

    def test_ocr_cli_writes_failure_records_including_repaired_but_damaged_pdfs(self):
        names = ("truncated", "encrypted", "zero_byte")
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_cli(*(str(ADVERSARIAL / f"{n}.pdf") for n in names), "--out", tmp,
                            module="pipeline.ocr_main")
            if "tesseract binary not usable" in proc.stderr:
                self.skipTest("tesseract not installed")
            self.assertEqual(proc.returncode, 1, proc.stderr[-2000:])
            for name in names:
                self._assert_failure_record(Path(tmp), name)
            record = json.loads((Path(tmp) / "truncated_status.json").read_text(encoding="utf-8"))
            self.assertEqual(record["error_class"], "DamagedPdfError")

    def test_a_different_pdf_with_the_same_name_is_kept_beside_the_original(self):
        import shutil

        from retrieval.registry import project

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_dir, pdf = tmp_dir / "out", tmp_dir / "kontrak.pdf"
            shutil.copy(SYNTHETIC / "bilingual_two_column.pdf", pdf)
            self.assertNotEqual(_run_cli(str(pdf), "--out", str(out_dir)).returncode, 1)
            original = json.loads((out_dir / "kontrak_raw.json").read_text(encoding="utf-8"))

            shutil.copy(SYNTHETIC / "private_parties_no_nip.pdf", pdf)
            for _ in range(2):
                self.assertNotEqual(_run_cli(str(pdf), "--out", str(out_dir)).returncode, 1)

            raws = sorted(p.name for p in out_dir.glob("*_raw.json"))
            self.assertEqual(len(raws), 2, raws)
            self.assertEqual(json.loads((out_dir / "kontrak_raw.json").read_text(encoding="utf-8"))["source"]["sha256"],
                             original["source"]["sha256"])
            revised_path = out_dir / next(r for r in raws if r != "kontrak_raw.json")
            revised = json.loads(revised_path.read_text(encoding="utf-8"))
            sha8 = revised["source"]["sha256"][:8]
            self.assertEqual(revised_path.name, f"kontrak__{sha8}_raw.json")
            self.assertEqual(project(revised).filename, f"kontrak ({sha8}).pdf")
            self.assertEqual(project(original).filename, "kontrak.pdf")

    def test_missing_file_writes_a_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_cli(str(ADVERSARIAL / "does_not_exist.pdf"), "--out", tmp)
            self.assertEqual(proc.returncode, 1)
            record = self._assert_failure_record(Path(tmp), "does_not_exist")
            self.assertEqual(record["error_class"], "FileNotFoundError")

    def test_failure_removes_a_stale_raw_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "truncated_raw.json"
            stale.write_text("{}", encoding="utf-8")
            proc = _run_cli(str(ADVERSARIAL / "truncated.pdf"), "--out", tmp)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self._assert_failure_record(Path(tmp), "truncated")

    def test_batch_continues_past_a_bad_file(self):
        good = [SYNTHETIC / "bilingual_two_column.pdf", SYNTHETIC / "private_parties_no_nip.pdf"]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            proc = _run_cli(str(good[0]), str(ADVERSARIAL / "truncated.pdf"), str(good[1]), "--out", tmp)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            for pdf in good:
                self.assertTrue((out_dir / f"{pdf.stem}_raw.json").exists(), proc.stdout + proc.stderr)
                self.assertFalse((out_dir / f"{pdf.stem}_status.json").exists())
            self._assert_failure_record(out_dir, "truncated")


if __name__ == "__main__":
    unittest.main()
