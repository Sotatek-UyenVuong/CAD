"""search_tool.py — Semantic search tool with optional image input.

Flow:
  text query (optional)
  + image bytes (optional)
    → Gemini Flash Vision: describe image in detail
    → enrich query: "<query> <description>"
  → Cohere embed (search_query, 1024-dim)
  → Qdrant top-K (filter by folder/file if provided)
  → Fetch page metadata from MongoDB
  → Return top-N results with rank, file_name, page_number, image_url,
    short_summary, vector_score
"""

from __future__ import annotations

import base64
from pathlib import Path

from cad_pipeline.config import (
    GEMINI_API_KEY,
    GEMINI_FLASH_MODEL,
    SIMILARITY_CUTOFF_VECTORSEARCH,
    TOP_K,
    TOP_N,
)
from cad_pipeline.core.embeddings import embed_query
from cad_pipeline.storage import mongo, qdrant_store


# ── Image description via Gemini Flash Vision ──────────────────────────────

def _describe_image(
    image_bytes: bytes,
    hint: str = "",
    model: str | None = None,
) -> str:
    """Ask Gemini Flash to describe an image for use as a search query.

    Returns a concise description string (1–3 sentences).
    Falls back to empty string on error.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types as _gt  # type: ignore

        _model = model or GEMINI_FLASH_MODEL
        client = genai.Client(api_key=GEMINI_API_KEY)

        hint_text = f'\nUser context: "{hint}"' if hint else ""
        prompt = (
            "You are analysing a CAD architectural drawing image for semantic search.\n"
            "Describe the image concisely: type of drawing (floor plan / elevation / "
            "section / detail / table / diagram), floor/level if visible, main elements "
            "(rooms, equipment, structures, symbols), and any notable labels or numbers."
            f"{hint_text}\n"
            "Reply in 1–3 sentences. Be specific. Output only the description, no preamble."
        )

        b64 = base64.standard_b64encode(image_bytes).decode()
        response = client.models.generate_content(
            model=_model,
            contents=[
                _gt.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        return response.text.strip()
    except Exception:
        return ""


# ── Core search logic ───────────────────────────────────────────────────────

def run_search_tool(
    query: str | None = None,
    image_bytes: bytes | None = None,
    image_path: str | Path | None = None,
    folder_id: str | None = None,
    file_id: str | None = None,
    top_n: int = 10,
    top_k: int | None = None,
    gemini_model: str | None = None,
) -> dict:
    """Semantic search across indexed pages.  Accepts text, image, or both.

    Args:
        query:        Natural language query (any language). Optional if image
                      is provided.
        image_bytes:  Raw image bytes (PNG / JPG / WebP). Takes precedence over
                      image_path.
        image_path:   Path to an image file. Used if image_bytes is None.
        folder_id:    Restrict search to a specific folder.
        file_id:      Restrict search to a specific file.
        top_n:        Max results to return (default 10).
        top_k:        Qdrant candidates (default TOP_K from config).
        gemini_model: Override Gemini model for image description.

    Returns:
        {
          "query_used":          str,   # final query sent to embedder
          "image_description":   str,   # Gemini's description (empty if no image)
          "total":               int,
          "results": [
            {
              "rank":           int,
              "file_id":        str,
              "file_name":      str,
              "page_number":    int,
              "image_url":      str,
              "short_summary":  str,    # first 200 chars of page short_summary
              "vector_score":   float,
            },
            ...
          ]
        }
    """
    # ── 1. Resolve image bytes ──────────────────────────────────────────────
    _img: bytes | None = image_bytes
    if _img is None and image_path:
        try:
            _img = Path(image_path).read_bytes()
        except Exception:
            _img = None

    # ── 2. Describe image with Gemini (if provided) ─────────────────────────
    description = ""
    if _img:
        description = _describe_image(_img, hint=query or "", model=gemini_model)

    # ── 3. Build final query ────────────────────────────────────────────────
    parts = [p for p in [query, description] if p]
    if not parts:
        return {
            "query_used": "",
            "image_description": "",
            "total": 0,
            "results": [],
            "error": "No query text or image provided.",
        }
    final_query = " ".join(parts)

    # ── 4. Embed ────────────────────────────────────────────────────────────
    query_vector = embed_query(final_query)

    # ── 5. Qdrant search ────────────────────────────────────────────────────
    _top_k = top_k or TOP_K
    candidates = qdrant_store.search_pages(
        query_vector=query_vector,
        top_k=_top_k,
        folder_id=folder_id,
        file_id=file_id,
    )
    candidates = [c for c in candidates if c["score"] >= SIMILARITY_CUTOFF_VECTORSEARCH]

    if not candidates:
        return {
            "query_used": final_query,
            "image_description": description,
            "total": 0,
            "results": [],
        }

    # ── 6. Fetch MongoDB metadata ───────────────────────────────────────────
    page_ids = [c["page_id"] for c in candidates]
    pages_data = {p["_id"]: p for p in mongo.get_pages_by_ids(page_ids)}

    # Batch-fetch folder names to avoid N+1 lookups
    folder_ids = {c["folder_id"] for c in candidates[:top_n]}
    folders_map: dict[str, str] = {}
    for fid in folder_ids:
        fol = mongo.get_folder(fid)
        folders_map[fid] = fol.get("name", fid) if fol else fid

    results: list[dict] = []
    for rank, c in enumerate(candidates[:top_n], start=1):
        page_doc  = pages_data.get(c["page_id"], {})
        file_doc  = mongo.get_file(c["file_id"]) or {}
        summary   = page_doc.get("short_summary") or c.get("short_summary", "")
        fol_id    = c.get("folder_id", file_doc.get("folder_id", ""))
        results.append({
            "rank":          rank,
            "file_id":       c["file_id"],
            "file_name":     file_doc.get("file_name") or file_doc.get("original_name", c["file_id"]),
            "folder_id":     fol_id,
            "folder_name":   folders_map.get(fol_id, fol_id),
            "page_number":   c["page_number"],
            "image_url":     page_doc.get("image_url", ""),
            "short_summary": summary[:200],
            "vector_score":  round(c["score"], 4),
        })

    return {
        "query_used":        final_query,
        "image_description": description,
        "total":             len(results),
        "results":           results,
    }
