"""upload_pipeline.py — Full upload & indexing pipeline.

Flow:
  File → S3 (original)
       → Images → S3 (pages)
       → Layout Detection
       → Per-block processing → S3 (block crops, optional)
       → Build Context (with S3 URLs)
       → Save MongoDB
       → Embedding → Save Qdrant
       → Build file/folder summaries

USE_S3=true  → upload original + pages + blocks to S3, store S3 URLs in Mongo
USE_S3=false → keep images local, store local paths in Mongo
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import re
import shutil
import cv2
from pathlib import Path
from typing import Callable

from cad_pipeline.config import API_BASE_URL, LOCAL_IMAGES_DIR, LOCAL_ORIGINALS_DIR, PDF_DPI, USE_S3
from cad_pipeline.core.pdf_to_images import pdf_to_page_images
from cad_pipeline.core.layout_detect import LayoutDetector
from cad_pipeline.core.marker_pdf import marker_ocr_pdf, CHUNK_THRESHOLD
from cad_pipeline.core.page_processor import process_page_blocks, generate_page_summary
from cad_pipeline.core.context_builder import (
    build_page_context,
    build_file_summary,
    build_folder_summary,
    generate_file_short_summary,
)
from cad_pipeline.core.embeddings import embed_text
from cad_pipeline.storage import mongo, qdrant_store


def run_upload_pipeline(
    file_path: str | Path,
    file_id: str | None = None,
    file_name: str | None = None,
    folder_id: str = "default",
    folder_name: str = "Default Folder",
    dpi: int = PDF_DPI,
    score_thr: float = 0.5,
    upload_blocks: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run the full upload and indexing pipeline for a file.

    Args:
        file_path: Path to the PDF, image, or document to index.
        file_id: Optional explicit file ID (auto-generated if None).
        file_name: Display name for the file (defaults to filename).
        folder_id: Which folder this file belongs to.
        folder_name: Display name for the folder.
        dpi: Resolution for PDF rendering.
        score_thr: Layout detection confidence threshold.
        upload_blocks: Whether to also upload individual block crops to S3.
        progress_callback: Optional callable(message) for progress updates.

    Returns:
        {"file_id", "folder_id", "total_pages", "page_ids", "original_url"}
    """
    file_path = Path(file_path)
    file_id = file_id or _make_id(str(file_path))
    file_name = file_name or file_path.name

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # ── Step 1: Upload original file to S3 or persist local original ───────
    original_url = str(file_path)
    if USE_S3:
        log(f"[1/8] Uploading original file to S3...")
        from cad_pipeline.storage.s3_store import upload_original_file
        original_url = upload_original_file(file_path, folder_id, file_id, file_name)
    else:
        log(f"[1/8] S3 disabled — persisting original file locally")
        safe_name = Path(file_name).name or file_path.name
        target_dir = LOCAL_ORIGINALS_DIR / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        src_resolved = file_path.resolve()
        dst_resolved = target_path.resolve()
        if src_resolved != dst_resolved:
            shutil.copy2(file_path, target_path)
        original_url = str(target_path)

    # ── Step 2: Render PDF → page images ────────────────────────────────────
    log(f"[2/8] Rendering {file_name} → page images (dpi={dpi})...")
    pages_info = pdf_to_page_images(file_path, file_id, LOCAL_IMAGES_DIR, dpi=dpi)
    total_pages = len(pages_info)

    # ── Step 2b: Marker whole-PDF OCR for large documents ───────────────────
    # For PDFs > CHUNK_THRESHOLD pages, submit the full PDF to Marker in
    # 10-page chunks (all concurrent). Results cached keyed by 1-based page
    # number. Table blocks later use this to skip per-crop Marker calls.
    marker_page_cache: dict[int, str] = {}
    if file_path.suffix.lower() == ".pdf" and total_pages > CHUNK_THRESHOLD:
        log(f"[2b/8] Large PDF ({total_pages} pages) — running Marker chunked OCR "
            f"({CHUNK_THRESHOLD}+ pages → chunks of 10)...")
        try:
            marker_page_cache = marker_ocr_pdf(file_path)
            log(f"[2b/8] Marker OCR complete — {len(marker_page_cache)} pages cached")
        except Exception as exc:
            log(f"[2b/8] Marker PDF OCR failed ({exc}) — falling back to per-crop mode")

    # ── Step 3: Ensure folder + file records in MongoDB ─────────────────────
    log(f"[3/8] Creating folder + file records in MongoDB...")
    mongo.upsert_folder(folder_id, folder_name)
    mongo.upsert_file(
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        file_url=original_url,
        total_pages=total_pages,
    )

    detector = LayoutDetector.get(score_thr=score_thr)
    page_summaries: list[str] = []
    page_ids: list[str] = []
    qdrant_records: list[dict] = []
    title_block_index: list[dict] = []

    for page_info in pages_info:
        page_number = page_info["page_number"]
        image_path = page_info["image_path"]

        log(f"[4/8] Processing page {page_number}/{total_pages}...")

        # ── Step 4a: Upload page image to S3 or build local HTTP URL ────────
        if USE_S3:
            from cad_pipeline.storage.s3_store import upload_page_image
            image_url = upload_page_image(image_path, folder_id, file_id, page_number)
        else:
            # Serve via FastAPI StaticFiles mount at /images
            rel = Path(image_path).relative_to(LOCAL_IMAGES_DIR)
            image_url = f"{API_BASE_URL}/images/{rel}"

        # ── Step 4b: Load image for processing ──────────────────────────────
        image = cv2.imread(str(image_path))

        # ── Step 5: Layout detection ─────────────────────────────────────────
        log(f"[5/8] Layout detection — page {page_number}...")
        blocks = detector.predict_file(image_path)

        # ── Step 6: Page summary + block processing — run concurrently ─────
        # marker_page_md: pre-cached from chunked PDF OCR (large docs only)
        marker_page_md = marker_page_cache.get(page_number)
        log(f"[6/8] Gemini: page summary + {len(blocks)} blocks in parallel — page {page_number}"
            + (" [Marker cache hit]" if marker_page_md else "") + "...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _page_executor:
            _summary_future = _page_executor.submit(generate_page_summary, image)
            _blocks_future = _page_executor.submit(
                process_page_blocks, blocks, image, marker_page_md
            )
            short_summary = _summary_future.result()
            processed_blocks = _blocks_future.result()
        page_summaries.append(short_summary)
        title_block_index.extend(_extract_title_block_entries(page_number, processed_blocks))

        # ── Step 6c: Upload block crops to S3 (optional) ─────────────────────
        if USE_S3 and upload_blocks:
            _upload_block_crops(
                blocks=blocks,
                image=image,
                processed_blocks=processed_blocks,
                folder_id=folder_id,
                file_id=file_id,
                page_number=page_number,
            )

        # ── Step 7: Build context_md ─────────────────────────────────────────
        context_md = build_page_context(
            page_number=page_number,
            image_url=image_url,
            short_summary=short_summary,
            blocks=processed_blocks,
        )

        page_id = f"{file_id}_p{page_number}"
        page_ids.append(page_id)

        # ── Step 7b: Save page to MongoDB ────────────────────────────────────
        mongo.upsert_page(
            page_id=page_id,
            file_id=file_id,
            folder_id=folder_id,
            page_number=page_number,
            image_url=image_url,
            short_summary=short_summary,
            context_md=context_md,
            blocks=processed_blocks,
        )

        # ── Embedding (Cohere, search_document) ──────────────────────────────
        vector = embed_text(short_summary, input_type="search_document")
        qdrant_records.append({
            "page_id": page_id,
            "vector": vector,
            "file_id": file_id,
            "folder_id": folder_id,
            "page_number": page_number,
            "short_summary": short_summary,
        })

    # ── Step 8: Batch upsert vectors to Qdrant ───────────────────────────────
    log(f"[8/8] Saving {len(qdrant_records)} vectors to Qdrant...")
    qdrant_store.upsert_page_vectors_batch(qdrant_records)

    # Build + save file-level summary (full, stored in DB for agent retrieval)
    file_summary = build_file_summary(page_summaries, file_name)
    mongo.update_file_summary(file_id, file_summary)

    # Generate a short summary for this file via Gemini Flash
    file_short_summary = generate_file_short_summary(file_name, page_summaries)
    mongo.update_file_short_summary(file_id, file_short_summary)
    mongo.update_file_title_block_index(file_id, title_block_index)

    # Rebuild folder-level summary using short summaries of all files in folder
    all_file_docs = mongo.list_files(folder_id)
    all_short_summaries = [f.get("short_summary") or f.get("summary", "") for f in all_file_docs if f.get("short_summary") or f.get("summary")]
    folder_summary = build_folder_summary(all_short_summaries)
    mongo.update_folder_summary(folder_id, folder_summary)

    log("✓ Done!")
    return {
        "file_id": file_id,
        "folder_id": folder_id,
        "total_pages": total_pages,
        "page_ids": page_ids,
        "original_url": original_url,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _upload_block_crops(
    blocks,
    image,
    processed_blocks: list[dict],
    folder_id: str,
    file_id: str,
    page_number: int,
) -> None:
    """Upload each block crop to S3 and attach the crop_url to processed_blocks."""
    import cv2 as _cv2
    from cad_pipeline.storage.s3_store import upload_block_crop

    for i, (block, proc) in enumerate(zip(blocks, processed_blocks)):
        crop = block.crop(image)
        success, buf = _cv2.imencode(".png", crop)
        if not success:
            continue
        crop_url = upload_block_crop(
            image_bytes=buf.tobytes(),
            folder_id=folder_id,
            file_id=file_id,
            page_number=page_number,
            block_type=block.label,
            block_index=i,
        )
        proc["crop_url"] = crop_url


def _make_id(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()[:12]


def _normalize_title_block_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_title_block_entries(page_number: int, processed_blocks: list[dict]) -> list[dict]:
    """Extract normalized title-block metadata from processed blocks for indexing."""
    entries: list[dict] = []
    for block in processed_blocks:
        if str(block.get("type", "")) != "title_block":
            continue
        content = block.get("content")
        if not isinstance(content, dict):
            continue
        drawing_no = _normalize_title_block_text(
            content.get("drawing_no")
            or content.get("drawing_number")
            or content.get("sheet_no")
            or content.get("sheet_number")
            or content.get("図面番号")
            or content.get("図番")
        )
        drawing_title = _normalize_title_block_text(
            content.get("drawing_title")
            or content.get("title")
            or content.get("sheet_title")
            or content.get("図面名称")
            or content.get("図面名")
        )
        project = _normalize_title_block_text(
            content.get("project")
            or content.get("project_name")
            or content.get("工事名")
            or content.get("物件名")
        )
        if not (drawing_no or drawing_title or project):
            continue
        entries.append(
            {
                "page_number": int(page_number),
                "drawing_no": drawing_no,
                "drawing_title": drawing_title,
                "project": project,
                "source": "title_block",
                "raw": content,
            }
        )
    return entries
