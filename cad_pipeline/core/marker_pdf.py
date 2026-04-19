"""marker_pdf.py — Chunked PDF submission to Marker API.

For PDFs > CHUNK_THRESHOLD pages, splits into CHUNK_SIZE-page sub-PDFs,
submits each chunk to Marker in parallel, then remaps Marker's 0-based
chunk-local page indices back to the original document's 1-based page numbers.

Usage:
    from cad_pipeline.core.marker_pdf import marker_ocr_pdf

    # Returns {page_number (1-based): markdown_string}
    page_markdowns = marker_ocr_pdf(Path("document.pdf"))

Integration with upload_pipeline:
    - Called once before the per-page loop when total_pages > CHUNK_THRESHOLD.
    - Results passed as `marker_page_md` to process_page_blocks per page.
    - Table blocks use the precomputed markdown and skip individual crop uploads.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import time
from pathlib import Path

import requests

from cad_pipeline.config import (
    MARKER_API_KEY,
    MARKER_API_URL,
    MARKER_MAX_POLLS,
    MARKER_POLL_INTERVAL,
)

CHUNK_THRESHOLD = 20   # split into chunks if PDF has more than this many pages
CHUNK_SIZE = 10        # pages per chunk


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def marker_ocr_pdf(
    pdf_path: Path,
    langs: str = "ja,en",
    mode: str = "fast",
) -> dict[int, str]:
    """OCR a PDF via Marker and return {page_number (1-based): markdown}.

    For PDFs ≤ CHUNK_THRESHOLD pages: submits the whole file in one request.
    For PDFs > CHUNK_THRESHOLD pages: splits into CHUNK_SIZE-page sub-PDFs,
    submits all chunks concurrently, and merges results with correct page
    number offsets (chunk pages reset to 0 after cutting → remapped here).

    Returns empty dict if MARKER_API_KEY is not configured.
    """
    if not MARKER_API_KEY:
        return {}

    try:
        import fitz  # noqa: F401 — verify PyMuPDF is available
    except ImportError as exc:
        raise ImportError("PyMuPDF required: pip install pymupdf") from exc

    import fitz

    doc = fitz.open(str(pdf_path))
    total = doc.page_count
    doc.close()

    if total <= CHUNK_THRESHOLD:
        # Single submission — no remapping needed
        chunks: list[tuple[int, bytes]] = [(0, pdf_path.read_bytes())]
    else:
        chunks = _split_into_chunks(pdf_path, CHUNK_SIZE)

    results: dict[int, str] = {}

    def _process(start_page: int, pdf_bytes: bytes) -> dict[int, str]:
        check_url = _submit_chunk(pdf_bytes, langs=langs, mode=mode)
        data = _poll_until_done(check_url)
        # offset converts chunk-local 0-based index → original 1-based page number
        return _extract_pages(data, offset=start_page)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fut_map = {ex.submit(_process, start, data): start for start, data in chunks}
        for fut in concurrent.futures.as_completed(fut_map):
            start = fut_map[fut]
            try:
                results.update(fut.result())
            except Exception as exc:
                # Store error string so pipeline can surface it
                results[start + 1] = f"[Marker chunk error (chunk_start={start}): {exc}]"

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PDF chunking
# ══════════════════════════════════════════════════════════════════════════════

def _split_into_chunks(pdf_path: Path, chunk_size: int) -> list[tuple[int, bytes]]:
    """Split PDF into (start_page_0based, pdf_bytes) tuples of `chunk_size` pages.

    Example for a 35-page PDF with chunk_size=10:
        (0,  pages 0–9)   → original pages 1–10
        (10, pages 10–19) → original pages 11–20
        (20, pages 20–29) → original pages 21–30
        (30, pages 30–34) → original pages 31–35
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    chunks: list[tuple[int, bytes]] = []

    for start in range(0, doc.page_count, chunk_size):
        end = min(start + chunk_size, doc.page_count)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        buf = io.BytesIO()
        chunk_doc.save(buf)
        chunk_doc.close()
        chunks.append((start, buf.getvalue()))

    doc.close()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Marker API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _submit_chunk(pdf_bytes: bytes, langs: str, mode: str) -> str:
    """POST a PDF chunk to Marker, return the check URL."""
    headers = {"X-API-Key": MARKER_API_KEY}
    payload = {
        "langs": langs,
        "force_ocr": "false",
        "paginate": "true",       # need per-page output for remapping
        "output_format": "json",  # JSON gives structured pages array
        "mode": mode,
        "disable_image_extraction": "true",
        "disable_ocr_math": "true",
        "use_llm": "false",
        "skip_cache": "false",
    }
    resp = requests.post(
        MARKER_API_URL,
        data=payload,
        files={"file": ("chunk.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    submit = resp.json()
    check_url = submit.get("request_check_url")
    if not check_url:
        raise RuntimeError(f"Marker returned no check_url: {submit}")
    return check_url


def _poll_until_done(check_url: str) -> dict:
    """Poll Marker until the job is complete and return the full response."""
    headers = {"X-API-Key": MARKER_API_KEY}
    for _ in range(MARKER_MAX_POLLS):
        time.sleep(MARKER_POLL_INTERVAL)
        poll = requests.get(check_url, headers=headers, timeout=30)
        data = poll.json()
        status = data.get("status")
        if status == "complete":
            return data
        if status == "error":
            raise RuntimeError(f"Marker processing error: {data.get('error')}")
    raise TimeoutError("Marker job did not complete within the polling window")


# ══════════════════════════════════════════════════════════════════════════════
# Page extraction & remapping
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pages(data: dict, offset: int) -> dict[int, str]:
    """Convert a Marker response to {original_1based_page_number: markdown}.

    offset: 0-based first-page index of this chunk in the original document.

    Marker JSON (paginate=true, output_format=json) structure:
        {"status": "complete", "json": {"pages": [{"markdown": "...", ...}, ...]}}
    Marker may also return a top-level "pages" list.

    Fallback: if no pages structure, parse form-feed or HR separators from
    the markdown field.
    """
    result: dict[int, str] = {}

    # ── Try structured JSON pages array ──────────────────────────────────────
    pages = (
        (data.get("json") or {}).get("pages")
        or data.get("pages")
    )
    if pages and isinstance(pages, list):
        for local_idx, page_obj in enumerate(pages):
            original_page = offset + local_idx + 1  # convert to 1-based
            result[original_page] = _page_obj_to_markdown(page_obj)
        return result

    # ── Fallback: parse paginated markdown ───────────────────────────────────
    markdown = (data.get("markdown") or "").strip()
    if not markdown:
        return result

    for local_idx, text in enumerate(_split_markdown_pages(markdown)):
        original_page = offset + local_idx + 1
        result[original_page] = text

    return result


def _page_obj_to_markdown(page: object) -> str:
    """Coerce a Marker page object to a plain markdown string."""
    if isinstance(page, str):
        return page
    if isinstance(page, dict):
        for key in ("markdown", "text", "content"):
            if key in page:
                return str(page[key])
        return json.dumps(page, ensure_ascii=False)
    return str(page)


def _split_markdown_pages(markdown: str) -> list[str]:
    """Split Marker's paginated markdown blob into per-page strings.

    Marker uses form-feed (\f) as the canonical page separator.
    Falls back to HR (\n---\n) if no form-feed is present.
    """
    if "\f" in markdown:
        pages = [p.strip() for p in markdown.split("\f") if p.strip()]
        return pages if pages else [markdown]

    parts = [p.strip() for p in markdown.split("\n---\n") if p.strip()]
    return parts if parts else [markdown]
