"""page_processor.py — Deep processing of each layout block per page.

OCR strategy (using Marker API — Datalab.to):
  - text, table  → Marker API (cloud OCR → Markdown output)

Gemini strategy (vision):
  - diagram      → Gemini Pro vision → description
  - title_block  → Gemini Flash vision → structured JSON metadata (no OCR step)

Marker API flow:
  POST /api/v1/marker  (multipart: file=crop.png + params)
  → {request_check_url}
  → GET check_url until status == "complete"
  → response.markdown
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image as PILImage

from cad_pipeline.config import (
    GEMINI_API_KEY,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    MARKER_API_KEY,
    MARKER_API_URL,
    MARKER_MAX_POLLS,
    MARKER_POLL_INTERVAL,
)
from cad_pipeline.core.layout_detect import LayoutBlock
from cad_pipeline.core.block_sorter import group_text_table_runs, sort_reading_order


# ══════════════════════════════════════════════════════════════════════════════
# Marker API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _crop_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a BGR numpy crop as PNG bytes."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _marker_ocr(
    image: np.ndarray,
    langs: str = "ja",
    mode: str = "fast",
    output_format: str = "markdown",
    use_llm: bool = False,
) -> str:
    """Send a cropped image to the Marker API and return the OCR result.

    Args:
        image: BGR numpy array (crop of the block).
        langs: Language hint, e.g. "ja", "ja,en".
        mode: "fast" or "high_quality".
        output_format: "markdown" or "json".
        use_llm: Whether to enable Marker's LLM enhancement (costs more).

    Returns:
        Markdown text extracted from the image, or an error string.
    """
    if not MARKER_API_KEY:
        return "[Marker API key not set — set MARKER_API_KEY in .env]"

    png_bytes = _crop_to_png_bytes(image)
    headers = {"X-API-Key": MARKER_API_KEY}

    payload = {
        "langs": langs,
        "force_ocr": "true",
        "format_lines": "false",
        "paginate": "false",
        "strip_existing_ocr": "false",
        "disable_image_extraction": "true",
        "disable_ocr_math": "true",
        "use_llm": str(use_llm).lower(),
        "mode": mode,
        "output_format": output_format,
        "skip_cache": "false",
    }

    # Submit job
    try:
        resp = requests.post(
            MARKER_API_URL,
            data=payload,
            files={"file": ("crop.png", png_bytes, "image/png")},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        submit = resp.json()
    except Exception as exc:
        return f"[Marker submit error: {exc}]"

    check_url = submit.get("request_check_url")
    if not check_url:
        return f"[Marker error: no check URL — {submit}]"

    # Poll until complete
    for _ in range(MARKER_MAX_POLLS):
        time.sleep(MARKER_POLL_INTERVAL)
        try:
            poll = requests.get(check_url, headers=headers, timeout=30)
            data = poll.json()
        except Exception as exc:
            return f"[Marker poll error: {exc}]"

        status = data.get("status")
        if status == "complete":
            if output_format == "markdown":
                return (data.get("markdown") or "").strip()
            return json.dumps(data.get("json") or {}, ensure_ascii=False)
        elif status == "error":
            return f"[Marker processing error: {data.get('error')}]"
        # Still processing — keep polling

    return "[Marker timeout: job did not complete in time]"


# ══════════════════════════════════════════════════════════════════════════════
# Gemini vision helpers (google-genai SDK)
# ══════════════════════════════════════════════════════════════════════════════

_gemini_client_instance = None


def _gemini_client():
    """Return a cached Gemini client (one per process — thread-safe for read)."""
    global _gemini_client_instance
    if _gemini_client_instance is None:
        from google import genai  # type: ignore
        _gemini_client_instance = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client_instance


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return buf.tobytes()


def _call_gemini_vision(model_name: str, prompt: str, image: np.ndarray) -> str:
    from google.genai import types  # type: ignore
    client = _gemini_client()
    png_bytes = _image_to_png_bytes(image)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
            prompt,
        ],
    )
    return response.text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Per-block processors
# ══════════════════════════════════════════════════════════════════════════════

def process_text_block(block: LayoutBlock, image: np.ndarray) -> dict:
    """Extract text via Gemini 2.5 Flash vision (Japanese + English)."""
    prompt = (
        "You are reading text from a CAD architectural drawing block. "
        "Extract ALL text exactly as it appears, preserving line breaks and layout. "
        "Do not translate or interpret — just transcribe. "
        "Reply with the raw text only, no commentary."
    )
    try:
        crop = block.crop(image)
        text = _call_gemini_vision(GEMINI_FLASH_MODEL, prompt, crop)
    except Exception as exc:
        text = f"[Text extraction error: {exc}]"

    return {
        "type": "text",
        "content": text,
        "bbox": list(block.bbox),
        "score": block.score,
    }


def process_table_block(
    block: LayoutBlock,
    image: np.ndarray,
    precomputed_md: str | None = None,
) -> dict:
    """Extract table via Marker API (fast mode) → Markdown table.

    If `precomputed_md` is provided (from a whole-page/PDF Marker OCR job),
    it is used directly, skipping the per-crop Marker call entirely.

    Fallback: if Marker returns empty or an error string, retry with Gemini
    Flash vision which handles complex multi-column Japanese tables better.
    """
    crop = block.crop(image)

    if precomputed_md is not None:
        # Pre-computed from chunked PDF OCR — use as-is
        markdown = precomputed_md
    else:
        markdown = _marker_ocr(crop, langs="ja,en", mode="fast")

    # Fallback to Gemini Flash when Marker returns nothing useful
    is_empty = not markdown.strip()
    is_error = markdown.startswith("[Marker")
    if is_empty or is_error:
        prompt = (
            "You are reading a table from a Japanese architectural CAD drawing. "
            "Extract ALL content from this table exactly as it appears. "
            "Return a clean Markdown table (with | separators). "
            "Preserve all Japanese text, numbers, and symbols precisely. "
            "Do not add commentary — only the Markdown table."
        )
        try:
            markdown = _call_gemini_vision(GEMINI_FLASH_MODEL, prompt, crop)
        except Exception as exc:
            markdown = f"[Table fallback error: {exc}]"

    return {
        "type": "table",
        "content": markdown,
        "bbox": list(block.bbox),
        "score": block.score,
    }


def process_diagram_block(block: LayoutBlock, image: np.ndarray) -> dict:
    """Gemini Pro vision — describe the diagram in detail."""
    prompt = (
        "You are analyzing a CAD architectural drawing block. "
        "Describe in detail what is shown in this diagram image: "
        "the type of drawing, objects, labels, dimensions, symbols visible. "
        "Be concise but complete. Reply in the same language as any text in the image."
    )
    try:
        crop = block.crop(image)
        description = _call_gemini_vision(GEMINI_PRO_MODEL, prompt, crop)
    except Exception as exc:
        description = f"[Diagram analysis error: {exc}]"

    return {
        "type": "diagram",
        "content": description,
        "bbox": list(block.bbox),
        "score": block.score,
    }


def process_title_block(block: LayoutBlock, image: np.ndarray) -> dict:
    """Gemini Flash vision — read the title block (表題欄) image directly and return structured JSON."""
    prompt = (
        "You are reading the title block (表題欄) of a Japanese architectural CAD drawing. "
        "Extract ALL structured information visible: project name, drawing number, scale, date, "
        "revision, company, engineer names, floor level, drawing title, and any other fields. "
        "Return ONLY a JSON object with English keys. "
        'Example: {"project": "...", "drawing_no": "...", "scale": "1:100", "date": "...", "revision": "..."}. '
        "No explanation, no markdown fences — just raw JSON."
    )
    crop = block.crop(image)
    try:
        raw = _call_gemini_vision(GEMINI_FLASH_MODEL, prompt, crop)
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        metadata = {"raw_text": raw}
    except Exception as exc:
        metadata = {"error": str(exc)}

    return {
        "type": "title_block",
        "content": metadata,
        "bbox": list(block.bbox),
        "score": block.score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

PROCESSORS = {
    "text": process_text_block,
    "table": process_table_block,
    "diagram": process_diagram_block,
    "title_block": process_title_block,
    "image": process_diagram_block,
}

# Maximum concurrent OCR / Gemini calls per page
_BLOCK_CONCURRENCY = 6

_TABLE_GROUP_PROMPT = (
    "You are reading content from a Japanese architectural CAD drawing. "
    "Extract ALL text and table content exactly as it appears. "
    "Return markdown tables (| separators) for tabular data, plain text for prose. "
    "Preserve all Japanese text, numbers, and symbols precisely. "
    "Do not add commentary."
)
_TEXT_GROUP_PROMPT = (
    "You are reading text from a CAD architectural drawing. "
    "Extract ALL text exactly as it appears, preserving line breaks and layout. "
    "Do not translate or interpret — just transcribe. "
    "Reply with the raw text only, no commentary."
)


def _merged_crop(group: list[dict], image: np.ndarray) -> np.ndarray:
    """Crop the merged bounding box of all blocks in a group."""
    x1 = int(min(b["bbox"][0] for b in group))
    y1 = int(min(b["bbox"][1] for b in group))
    x2 = int(max(b["bbox"][2] for b in group))
    y2 = int(max(b["bbox"][3] for b in group))
    h, w = image.shape[:2]
    return image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def _merged_bbox(group: list[dict]) -> list[int]:
    return [
        int(min(b["bbox"][0] for b in group)),
        int(min(b["bbox"][1] for b in group)),
        int(max(b["bbox"][2] for b in group)),
        int(max(b["bbox"][3] for b in group)),
    ]


def _process_solo(block_dict: dict, image: np.ndarray,
                  marker_page_md: str | None = None) -> dict:
    """Process a single LayoutBlock (solo group)."""
    block: LayoutBlock = block_dict["_lb"]
    if block.label == "table":
        return process_table_block(block, image, precomputed_md=marker_page_md)
    processor = PROCESSORS.get(block.label, process_text_block)
    return processor(block, image)


def _process_merged_group(group: list[dict], image: np.ndarray,
                           marker_page_md: str | None = None) -> dict:
    """Merge bboxes → single crop → single OCR/Gemini call.

    Mixed text+table groups use Marker (fallback Gemini Flash).
    Pure-text groups use Gemini Flash directly.
    """
    types = {b["type"] for b in group}
    crop = _merged_crop(group, image)
    bbox = _merged_bbox(group)
    score = max(b["score"] for b in group)

    if "table" in types:
        # Use Marker first; fall back to Gemini Flash if empty/error
        markdown = marker_page_md if marker_page_md else _marker_ocr(crop, langs="ja,en", mode="fast")
        if not markdown.strip() or markdown.startswith("[Marker"):
            try:
                markdown = _call_gemini_vision(GEMINI_FLASH_MODEL, _TABLE_GROUP_PROMPT, crop)
            except Exception as exc:
                markdown = f"[Merged group OCR error: {exc}]"
        dominant_type = "table"
    else:
        # Pure text group → Gemini Flash
        try:
            markdown = _call_gemini_vision(GEMINI_FLASH_MODEL, _TEXT_GROUP_PROMPT, crop)
        except Exception as exc:
            markdown = f"[Merged group text error: {exc}]"
        dominant_type = "text"

    return {
        "type": dominant_type,
        "content": markdown,
        "bbox": bbox,
        "score": score,
        "_merged_count": len(group),
    }


def process_page_blocks(
    blocks: list[LayoutBlock],
    image: np.ndarray,
    marker_page_md: str | None = None,
) -> list[dict]:
    """Sort blocks into reading order, group adjacent text/table, then OCR.

    Flow:
      1. Convert LayoutBlocks → dicts (preserving _lb reference).
      2. sort_reading_order  — column-major, left→right, top→bottom.
      3. group_text_table_runs — merge vertically adjacent text/table in
         the same column into single groups.
      4. For each group: if merged → merged-bbox crop → single OCR/Gemini;
         if solo → per-block processor as before.
      5. All groups processed concurrently (ThreadPoolExecutor).

    Returns a list of processed dicts in reading order (one per group).
    marker_page_md: pre-cached Marker OCR from chunked PDF (large docs).
    """
    if not blocks:
        return []

    # Attach LayoutBlock reference so _process_solo can use it
    block_dicts = [
        {"type": b.label, "bbox": list(b.bbox), "score": b.score, "_lb": b}
        for b in blocks
    ]

    sorted_dicts = sort_reading_order(block_dicts)
    groups = group_text_table_runs(sorted_dicts)

    results: list[dict] = [{}] * len(groups)

    def _process_group(idx: int, group: list[dict]) -> tuple[int, dict]:
        try:
            if len(group) == 1:
                return idx, _process_solo(group[0], image, marker_page_md)
            return idx, _process_merged_group(group, image, marker_page_md)
        except Exception as exc:
            bbox = _merged_bbox(group) if len(group) > 1 else list(group[0]["bbox"])
            return idx, {
                "type": group[0]["type"],
                "content": f"[Group processing error: {exc}]",
                "bbox": bbox,
                "score": group[0]["score"],
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=_BLOCK_CONCURRENCY) as executor:
        futures = [executor.submit(_process_group, i, g) for i, g in enumerate(groups)]
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Page short summary via Gemini Flash
# ══════════════════════════════════════════════════════════════════════════════

def generate_page_summary(image: np.ndarray) -> str:
    """Ask Gemini Flash for a 1–2 sentence summary of the full page image."""
    prompt = (
        "You are analyzing a page from a Japanese architectural CAD drawing. "
        "Write a 1–2 sentence summary describing what this page shows "
        "(drawing type, floor level, systems, key elements). "
        "Reply in Japanese if the drawing text is Japanese, otherwise in English."
    )
    try:
        return _call_gemini_vision(GEMINI_FLASH_MODEL, prompt, image)
    except Exception as exc:
        return f"[Summary error: {exc}]"
