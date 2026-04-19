"""folder_agent.py — Level 1 agent: decides if a query can be answered from folder/file summaries.

Input:
  - list of file short summaries + file_ids

Output:
  - {"action": "answer", "answer": "...", "file_ids": []}
  - {"action": "go_to_file", "answer": "", "file_ids": ["id1", "id2"]}
"""

from __future__ import annotations

import json
import re

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
from cad_pipeline.agents.language_utils import detect_query_language, language_label


def run_folder_agent(
    query: str,
    files: list[dict],
) -> dict:
    """Run the folder-level agent.

    Args:
        query: User's question.
        files: List of {"file_id": str, "file_name": str, "summary": str(short summary)}.

    Returns:
        {"action": "answer"|"go_to_file", "answer": str, "file_ids": list[str]}
    """
    from google import genai  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)
    lang = language_label(detect_query_language(query))

    files_text = "\n".join(
        f"- file_id={f['file_id']} | name={f.get('file_name','?')} | summary={f.get('summary','')[:300]}"
        for f in files
    )

    prompt = f"""You are a Folder-level assistant for a CAD document Q&A system.

Available files (with short summaries):
{files_text}

User question: "{query}"
Detected user language: {lang}

Your tasks:
1. Decide if this question can be answered using the summaries above
2. If YES → answer directly
3. If NO → identify the most relevant file(s) and escalate

Rules:
- ONLY answer directly for very high-level overview questions (e.g. "what files are in this folder?", "what is this project about?")
- For ANY question requiring specific numbers, counts, detailed specs, page content, or technical details → ALWAYS escalate (go_to_file)
- Do NOT guess or infer details not explicitly stated in the summaries
- Select up to 3 most relevant files when escalating
- The "answer" field MUST be written in {lang} only.

Reply ONLY as JSON:
{{
  "action": "answer" or "go_to_file",
  "answer": "<answer text, empty if go_to_file>",
  "file_ids": ["<file_id>", ...],
  "reason": "<why you chose this action>"
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
            "action": "go_to_file",
            "answer": "",
            "file_ids": [f["file_id"] for f in files[:2]],
            "reason": f"Router error: {exc}",
        }
