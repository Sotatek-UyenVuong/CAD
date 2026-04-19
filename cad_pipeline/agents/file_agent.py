"""file_agent.py — Level 2 agent: decides if file summary is sufficient or escalates to pages."""

from __future__ import annotations

import json
import re

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
from cad_pipeline.agents.language_utils import detect_query_language, language_label


def run_file_agent(
    query: str,
    file_id: str,
    file_name: str,
    file_short_summary: str,
    file_summary: str,
) -> dict:
    """Run the file-level agent.

    Args:
        query: User's question.
        file_id: MongoDB file ID.
        file_name: Human-readable file name.
        file_short_summary: Short file-level overview.
        file_summary: Detailed file-level summary (aggregated from pages).

    Returns:
        {"action": "answer"|"go_to_page", "answer": str, "reason": str}
    """
    from google import genai  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)
    lang = language_label(detect_query_language(query))

    prompt = f"""You are a File-level assistant for a CAD document Q&A system.

File: {file_name} (id={file_id})
Short summary (quick overview):
{file_short_summary}

Detailed summary (longer file context):
{file_summary}

User question: "{query}"
Detected user language: {lang}

Your tasks:
1. Decide if the file summaries are sufficient to answer the question
2. If YES → answer directly
3. If NO → escalate to page-level analysis

Rules:
- Use short summary for quick intent matching and detailed summary for content validation
- ONLY answer directly if the summaries EXPLICITLY contain the exact information needed
- For ANY question about specific numbers, counts, technical specs, detailed content, drawings, measurements → ALWAYS escalate (go_to_page)
- Do NOT infer or guess — if uncertain, escalate
- The summary is an overview only; detailed answers require page-level reading
- The "answer" field MUST be written in {lang} only.

Reply ONLY as JSON:
{{
  "action": "answer" or "go_to_page",
  "answer": "<answer text, empty if go_to_page>",
  "reason": "<brief explanation>"
}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        return result
    except Exception as exc:
        return {
            "action": "go_to_page",
            "answer": "",
            "reason": f"Agent error: {exc}",
        }
