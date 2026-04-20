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
        {
          "action": "answer"|"go_to_page",
          "answer": str,
          "reason": str,
          "candidate_pages": list[int]
        }
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
- If action is "go_to_page", "candidate_pages" MUST contain the most likely page numbers (up to 8).
- candidate_pages must be explicit integers in ascending order (example: [3, 7, 12]).
- If no reliable page candidates are visible from summary, return [].
- The "answer" field MUST be written in {lang} only.

Reply ONLY as JSON:
{{
  "action": "answer" or "go_to_page",
  "answer": "<answer text, empty if go_to_page>",
  "reason": "<brief explanation>",
  "candidate_pages": [<int>, ...]
}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Invalid file agent result format")

        action = str(result.get("action", "go_to_page"))
        answer = str(result.get("answer", ""))
        reason = str(result.get("reason", ""))

        candidate_pages_raw = result.get("candidate_pages", [])
        candidate_pages: list[int] = []
        if isinstance(candidate_pages_raw, list):
            seen: set[int] = set()
            for item in candidate_pages_raw:
                try:
                    page_no = int(item)
                except (TypeError, ValueError):
                    continue
                if page_no <= 0 or page_no in seen:
                    continue
                seen.add(page_no)
                candidate_pages.append(page_no)
        candidate_pages.sort()

        if action == "answer":
            candidate_pages = []

        return {
            "action": action,
            "answer": answer,
            "reason": reason,
            "candidate_pages": candidate_pages,
        }
    except Exception as exc:
        return {
            "action": "go_to_page",
            "answer": "",
            "reason": f"Agent error: {exc}",
            "candidate_pages": [],
        }
