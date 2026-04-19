"""upload_standard.py — compatibility wrapper for standard upload pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from cad_pipeline.pipeline.upload_pipeline import run_upload_pipeline


def run_upload_standard(
    file_path: Path | str,
    file_id: str | None = None,
    file_name: str | None = None,
    folder_id: str = "default",
    folder_name: str = "Default Folder",
    dpi: int = 300,
    score_thr: float = 0.5,
    upload_blocks: bool = False,
    checkpoint_path: Path | str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Forward to `run_upload_pipeline`.

    `checkpoint_path` is kept for backward compatibility with old callers.
    """
    _ = checkpoint_path
    return run_upload_pipeline(
        file_path=file_path,
        file_id=file_id,
        file_name=file_name,
        folder_id=folder_id,
        folder_name=folder_name,
        dpi=dpi,
        score_thr=score_thr,
        upload_blocks=upload_blocks,
        progress_callback=progress_callback,
    )
