#!/usr/bin/env python3
"""batch_upload_tae.py — Batch upload TAE drawings + additional PDF files.

Checkpoint/resume:
  - A master checkpoint at CHECKPOINT_DIR/master.json tracks which FILES are done
  - Each file also has its own per-page checkpoint inside upload_rendered.py
  - Re-running this script will skip already-completed files and continue interrupted ones

Usage:
    cd /mnt/data8tb/notex/uyenvuong/CAD
    python batch_upload_tae.py

    # Resume after interruption — just re-run the same command:
    python batch_upload_tae.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_pipeline.pipeline.upload_rendered import run_upload_rendered

# ── Paths ─────────────────────────────────────────────────────────────────────
TAE_PDF_DIR = PROJECT_ROOT / "cad_pipeline" / "TAE" / "260316_新綱島スクエア竣工図PDF"
PDF_DIR     = PROJECT_ROOT / "pdf"          # general PDF dir
RENDERED_ROOT = PROJECT_ROOT
DXF_ROOT = PROJECT_ROOT / "dxf_output"
CHECKPOINT_DIR = PROJECT_ROOT / "cad_pipeline" / ".checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_CHECKPOINT = CHECKPOINT_DIR / "master.json"

# ── Upload manifest ────────────────────────────────────────────────────────────
# Each entry: (pdf_path, rendered_dir, dxf_dir_or_None, folder_id, folder_name)
UPLOAD_MANIFEST = [
    # ── TAE 竣工図 (4 files) ──────────────────────────────────────────────────
    (
        TAE_PDF_DIR / "竣工図（新綱島スクエア　建築意匠図）.pdf",
        RENDERED_ROOT / "rendered_竣工図（新綱島スクエア　建築意匠図）_dpi300",
        DXF_ROOT / "竣工図（新綱島スクエア　建築意匠図）",
        "tae_architectural",
        "竣工図 — 建築意匠図",
    ),
    (
        TAE_PDF_DIR / "竣工図（新綱島スクエア　構造図）.pdf",
        RENDERED_ROOT / "rendered_竣工図（新綱島スクエア　構造図）_dpi300",
        DXF_ROOT / "竣工図（新綱島スクエア　構造図）",
        "tae_structural",
        "竣工図 — 構造図",
    ),
    (
        TAE_PDF_DIR / "竣工図（新綱島スクエア　機械設備図）.pdf",
        RENDERED_ROOT / "rendered_竣工図（新綱島スクエア　機械設備図）_dpi300",
        DXF_ROOT / "竣工図（新綱島スクエア　機械設備図）",
        "tae_mechanical",
        "竣工図 — 機械設備図",
    ),
    (
        TAE_PDF_DIR / "竣工図（新綱島スクエア　電気設備図）.pdf",
        RENDERED_ROOT / "rendered_竣工図（新綱島スクエア　電気設備図）_dpi300",
        DXF_ROOT / "竣工図（新綱島スクエア　電気設備図）",
        "tae_electrical",
        "竣工図 — 電気設備図",
    ),
    # ── Additional PDFs ───────────────────────────────────────────────────────
    (
        PDF_DIR / "2024入社説明会(第1設計統括室_住宅).pdf",
        RENDERED_ROOT / "rendered_2024入社説明会(第1設計統括室_住宅)_dpi300",
        None,
        "nyusha_2024",
        "2024入社説明会（第1設計統括室・住宅）",
    ),
    (
        PDF_DIR / "TLC_BZ商品計画テキスト2014.pdf",
        RENDERED_ROOT / "rendered_TLC_BZ商品計画テキスト2014_dpi300",
        None,
        "tlc_bz_2014",
        "TLC_BZ商品計画テキスト2014",
    ),
    (
        PDF_DIR / "総合設計制度の手引き・事例集.pdf",
        RENDERED_ROOT / "rendered_総合設計制度の手引き・事例集_dpi300",
        None,
        "sogo_sekkei",
        "総合設計制度の手引き・事例集",
    ),
]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    master = load_master(MASTER_CHECKPOINT)
    files_status: dict = master.setdefault("files", {})

    total = len(UPLOAD_MANIFEST)
    print(f"\n{'='*60}")
    print(f"TAE Batch Upload — {total} files")
    print(f"Checkpoint: {MASTER_CHECKPOINT}")
    print(f"{'='*60}\n")

    for idx, (pdf_path, rendered_dir, dxf_dir, folder_id, folder_name) in enumerate(UPLOAD_MANIFEST, 1):
        pdf_path     = Path(pdf_path)
        rendered_dir = Path(rendered_dir)
        dxf_dir      = Path(dxf_dir) if dxf_dir else None
        key          = pdf_path.name   # use filename as checkpoint key

        print(f"\n[{idx}/{total}] {'─'*50}")
        print(f"  PDF:      {pdf_path.name}")
        print(f"  Rendered: {rendered_dir.name}")
        print(f"  DXF dir:  {dxf_dir.name if dxf_dir else 'None'}")
        print(f"  Folder:   {folder_id} / {folder_name}")

        if files_status.get(key) == "done":
            print(f"  ✓ Already completed — skipping")
            continue

        if not pdf_path.exists():
            print(f"  ✗ PDF not found: {pdf_path} — skipping")
            files_status[key] = "skipped_missing_pdf"
            save_master(MASTER_CHECKPOINT, master)
            continue

        if not rendered_dir.exists():
            print(f"  ✗ Rendered dir not found: {rendered_dir} — skipping")
            files_status[key] = "skipped_missing_rendered"
            save_master(MASTER_CHECKPOINT, master)
            continue

        if dxf_dir and not dxf_dir.exists():
            print(f"  ⚠ DXF dir not found: {dxf_dir} — uploading without DXF mapping")
            dxf_dir = None

        # Per-file checkpoint inside .checkpoints/
        file_checkpoint = CHECKPOINT_DIR / f"{folder_id}_pages.json"

        files_status[key] = "in_progress"
        files_status[f"{key}_started_at"] = datetime.utcnow().isoformat()
        save_master(MASTER_CHECKPOINT, master)

        t0 = time.time()
        try:
            result = run_upload_rendered(
                pdf_path=pdf_path,
                rendered_dir=rendered_dir,
                dxf_dir=dxf_dir,
                folder_id=folder_id,
                folder_name=folder_name,
                checkpoint_path=file_checkpoint,
            )
            elapsed = time.time() - t0
            files_status[key] = "done"
            files_status[f"{key}_finished_at"] = datetime.utcnow().isoformat()
            files_status[f"{key}_elapsed_s"] = round(elapsed, 1)
            files_status[f"{key}_total_pages"] = result["total_pages"]
            save_master(MASTER_CHECKPOINT, master)
            print(f"\n  ✓ {key} complete — {result['total_pages']} pages in {elapsed:.0f}s")

        except KeyboardInterrupt:
            print(f"\n  ⚠ Interrupted! Progress saved to {file_checkpoint}")
            print("  Re-run the script to resume from the last completed page.")
            save_master(MASTER_CHECKPOINT, master)
            sys.exit(0)

        except Exception:
            elapsed = time.time() - t0
            files_status[key] = "error"
            files_status[f"{key}_error"] = traceback.format_exc()
            save_master(MASTER_CHECKPOINT, master)
            print(f"\n  ✗ ERROR processing {key} after {elapsed:.0f}s:")
            traceback.print_exc()
            print("  Continuing with next file...")

    print(f"\n{'='*60}")
    done = sum(1 for k, v in files_status.items() if not k.endswith(("_at", "_s", "_pages", "_error")) and v == "done")
    print(f"Batch complete: {done}/{total} files done")
    print(f"Master checkpoint: {MASTER_CHECKPOINT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
