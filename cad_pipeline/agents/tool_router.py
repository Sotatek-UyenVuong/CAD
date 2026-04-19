"""tool_router.py — Classify user intent to select the right tool.

Given a text query (optional) + an image (optional) + a hint about available
page context, Gemini Flash returns the tool that best matches the user's intent.

Tool catalogue
--------------
search       Find pages/documents similar to the image or description.
count        Count symbols, elements, or objects (text or visual).
area         Calculate floor area or room size (text or visual).
viz          Visualize / highlight elements on a drawing page.
report_pdf   Generate a structured PDF report from the current context.
report_excel Generate an Excel spreadsheet from the current context.
none         General Q&A — no specialist tool needed.
"""

from __future__ import annotations

import json
import re

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

# ── Gemini Flash classifier ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a tool-routing assistant for a CAD architectural drawing Q&A system.
Given a user's message (text and/or image), decide which specialist tool to use.

Available tools:
  search       — find pages/documents similar to this image or description
  count        — count symbols, elements, or objects in a drawing
  area         — calculate floor area or room sizes
  viz          — visualize or highlight elements on a drawing page
  report_pdf   — generate a structured PDF report from current context
  report_excel — generate an Excel spreadsheet from current context
  none         — general text Q&A (no specialist tool needed)

Rules:
- If ONLY an image is provided with no text → "search"
- If query asks "how many pages/drawings/files/documents/lists" (metadata counting) → "none"
- If query mentions counting symbols/equipment/objects in drawing content → "count"
- If query mentions area/m²/diện tích → "area"  
- If query mentions PDF/báo cáo/report → "report_pdf"
- If query mentions Excel/spreadsheet/bảng tính (including typo like "excell") → "report_excel"
- If query asks to visualize/highlight → "viz"
- If query asks to find similar drawings or pages → "search"
- Otherwise → "none"

Reply ONLY as JSON: {"tool": "<tool_name>", "reason": "<one sentence>"}
"""


def classify_tool(
    query: str | None = None,
    image_bytes: bytes | None = None,
    model: str | None = None,
) -> dict:
    """Classify user intent → tool name.

    Returns {"tool": str, "reason": str, "source": "gemini"|"default"}.
    Always returns a valid result (never raises).
    """
    # Default when no input at all
    if not query and not image_bytes:
        return {"tool": "none", "reason": "No input provided.", "source": "default"}

    # Image only → search
    if image_bytes and not query:
        return {"tool": "search", "reason": "Image provided without text → semantic search.", "source": "default"}

    # Deterministic guardrails before LLM routing:
    # "How many documents/pages/drawings" is metadata counting, not symbol counting.
    if query:
        q = query.lower()
        wants_count = any(k in q for k in ["bao nhiêu", "how many", "何個", "いくつ", "số lượng"])
        metadata_targets = [
            "bảng vẽ", "ban ve", "drawing", "drawings", "sheet", "sheets",
            "trang", "page", "pages", "tài liệu", "tai lieu", "document", "documents",
            "file", "files", "list", "danh sách", "一覧", "図面リスト",
        ]
        symbol_targets = [
            "symbol", "ký hiệu", "ki hieu", "block", "text", "insert", "thiết bị", "thiet bi",
            "đèn", "den", "công tắc", "cong tac", "ổ cắm", "o cam", "van", "valve",
        ]
        if wants_count and any(k in q for k in metadata_targets) and not any(k in q for k in symbol_targets):
            return {
                "tool": "none",
                "reason": "Metadata counting query (documents/pages/drawings), not symbol counting.",
                "source": "default",
            }

    # Gemini Flash for ambiguous or image+text cases
    try:
        from google import genai  # type: ignore
        from google.genai import types as _gt  # type: ignore

        _model = model or GEMINI_FLASH_MODEL
        client = genai.Client(api_key=GEMINI_API_KEY)

        contents: list = [_SYSTEM_PROMPT]
        if image_bytes:
            contents.append(_gt.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        if query:
            contents.append(f'User message: "{query}"')

        response = client.models.generate_content(model=_model, contents=contents)
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)
        tool = parsed.get("tool", "none")
        # Validate
        valid = {"search", "count", "area", "viz", "report_pdf", "report_excel", "none"}
        if tool not in valid:
            tool = "none"
        return {"tool": tool, "reason": parsed.get("reason", ""), "source": "gemini"}

    except Exception as exc:
        return {
            "tool": "none",
            "reason": f"Gemini unavailable ({exc}); fallback to none.",
            "source": "default",
        }
