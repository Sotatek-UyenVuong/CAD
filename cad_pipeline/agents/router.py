"""router.py — Classify user query as Q&A or Search before routing to agents."""

from __future__ import annotations

import json
import re

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

_SEARCH_KEYWORDS = [
    "tìm", "search", "ảnh này", "giống", "find similar", "similar to",
    "find image", "find diagram", "tìm kiếm", "検索", "似ている", "探す",
]


def classify_query(query: str) -> str:
    """Classify a user query into 'qa' or 'search'.

    Uses keyword matching first (fast path), then falls back to Gemini.

    Returns:
        'qa' or 'search'
    """
    q_lower = query.lower()
    if any(kw in q_lower for kw in _SEARCH_KEYWORDS):
        return "search"

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""Classify this user query into one of two types:
1. "qa" → question answering about document content (quantities, areas, descriptions, specs)
2. "search" → searching for images, diagrams, or visually similar content

Query: "{query}"

Rules:
- If query contains "find similar", "image like", "diagram like", "search for image" → search
- Otherwise → qa

Reply ONLY as JSON: {{"type": "qa"}} or {{"type": "search"}}"""

        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        return result.get("type", "qa")
    except Exception:
        return "qa"
