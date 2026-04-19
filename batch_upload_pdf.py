#!/usr/bin/env python3
"""batch_upload_pdf.py — Batch upload PDFs từ /pdf/ qua luồng chuẩn (layout detect).

Chỉ upload những file chưa có pre-rendered dir → dùng upload_standard.py
(render PDF + layout detect + OCR/Gemini song song N pages).

Checkpoint/resume:
  - master.json tracks file-level status
  - Re-run để resume file bị ngắt giữa chừng
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_pipeline.pipeline.upload_pipeline import run_upload_pipeline

PDF_DIR        = PROJECT_ROOT / "pdf"
RENDERED_ROOT  = PROJECT_ROOT
CHECKPOINT_DIR = PROJECT_ROOT / "cad_pipeline" / ".checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_CHECKPOINT = CHECKPOINT_DIR / "master_pdf.json"


def _folder_id(pdf_name: str) -> str:
    """Make a safe folder_id from a PDF filename."""
    slug = re.sub(r'[^\w\-]', '_', Path(pdf_name).stem, flags=re.UNICODE)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:48].lower()


def load_master(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"files": {}}


def save_master(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main() -> None:
    # Collect PDFs without pre-rendered dirs
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    manifest = []
    for pdf in all_pdfs:
        rendered = RENDERED_ROOT / f"rendered_{pdf.stem}_dpi300"
        if not rendered.exists():
            folder_id   = _folder_id(pdf.name)
            folder_name = pdf.stem
            manifest.append((pdf, folder_id, folder_name))

    total = len(manifest)
    print(f"\n{'='*60}")
    print(f"PDF Batch Upload — {total} files (standard pipeline)")
    print(f"Checkpoint: {MASTER_CHECKPOINT}")
    print(f"{'='*60}\n")

    master = load_master(MASTER_CHECKPOINT)
    files_status: dict = master.setdefault("files", {})

    for idx, (pdf_path, folder_id, folder_name) in enumerate(manifest, 1):
        key = pdf_path.name
        print(f"\n[{idx}/{total}] {'─'*50}")
        print(f"  PDF:    {key}")
        print(f"  Folder: {folder_id} / {folder_name}")

        if files_status.get(key) == "done":
            print(f"  ✓ Already completed — skipping")
            continue

        file_checkpoint = CHECKPOINT_DIR / f"{folder_id}_std_pages.json"

        files_status[key] = "in_progress"
        files_status[f"{key}_started_at"] = datetime.utcnow().isoformat()
        save_master(MASTER_CHECKPOINT, master)

        t0 = time.time()
        try:
            result = run_upload_pipeline(
                file_path=pdf_path,
                folder_id=folder_id,
                folder_name=folder_name,
                checkpoint_path=file_checkpoint,
            )
            elapsed = time.time() - t0
            files_status[key] = "done"
            files_status[f"{key}_finished_at"] = datetime.utcnow().isoformat()
            files_status[f"{key}_elapsed_s"]   = round(elapsed, 1)
            files_status[f"{key}_total_pages"] = result["total_pages"]
            save_master(MASTER_CHECKPOINT, master)
            print(f"\n  ✓ {key} — {result['total_pages']} pages in {elapsed:.0f}s")

        except KeyboardInterrupt:
            print(f"\n  ⚠ Interrupted! Progress saved to {file_checkpoint}")
            print("  Re-run to resume.")
            save_master(MASTER_CHECKPOINT, master)
            sys.exit(0)

        except Exception:
            elapsed = time.time() - t0
            files_status[key] = "error"
            files_status[f"{key}_error"] = traceback.format_exc()
            save_master(MASTER_CHECKPOINT, master)
            print(f"\n  ✗ ERROR: {key} after {elapsed:.0f}s:")
            traceback.print_exc()
            print("  Continuing with next file...")

    done = sum(1 for k, v in files_status.items()
               if not any(k.endswith(s) for s in ('_at','_s','_pages','_error'))
               and v == "done")
    print(f"\n{'='*60}")
    print(f"Batch complete: {done}/{total} files done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
