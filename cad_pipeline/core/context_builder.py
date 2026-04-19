"""context_builder.py — Build a clean Markdown context string from processed blocks.

Output format per page:

# Page {N}
![page image]({image_url})

## Summary
{short_summary}

## Content

### [TEXT]
{text}

### [TABLE]
{markdown_table}

### [DIAGRAM]
Description: {description}

### [TITLE_BLOCK]
{json metadata}
"""

from __future__ import annotations

import json


def build_page_context(
    page_number: int,
    image_url: str,
    short_summary: str,
    blocks: list[dict],
) -> str:
    """Build the context_md string for a single page.

    Blocks arrive pre-sorted in reading order and pre-merged by
    process_page_blocks (sort_reading_order + group_text_table_runs ran
    before OCR). Each dict may have _merged_count > 1 if it represents a
    group of adjacent blocks that were OCR'd together as a single crop.

    Args:
        page_number: 1-based page index.
        image_url: URL or local path to the full-page image.
        short_summary: 1–2 sentence summary produced by Gemini Flash.
        blocks: List of processed block dicts, already in reading order.

    Returns:
        A clean Markdown string ready to be stored in MongoDB.
    """
    lines: list[str] = []

    lines.append(f"# Page {page_number}")
    lines.append(f"![page image]({image_url})")
    lines.append("")
    lines.append("## Summary")
    lines.append(short_summary or "")
    lines.append("")
    lines.append("## Content")
    lines.append("")

    for block in blocks:
        _render_block(block, lines)

    return "\n".join(lines)


def _render_block(block: dict, lines: list[str], in_group: bool = False) -> None:
    """Append block content to lines list."""
    block_type = block.get("type", "text").upper()
    content = block.get("content", "")

    if block_type == "TEXT":
        if content and not content.startswith("["):
            if not in_group:
                lines.append("### [TEXT]")
            lines.append(content)
            if not in_group:
                lines.append("")

    elif block_type == "TABLE":
        if content:
            if not in_group:
                lines.append("### [TABLE]")
            lines.append(content)
            if not in_group:
                lines.append("")

    elif block_type == "DIAGRAM":
        lines.append("### [DIAGRAM]")
        if isinstance(content, str):
            lines.append(f"Description: {content}")
        lines.append("")

    elif block_type == "IMAGE":
        lines.append("### [IMAGE]")
        if isinstance(content, str):
            lines.append(f"Description: {content}")
        lines.append("")

    elif block_type == "TITLE_BLOCK":
        lines.append("### [TITLE_BLOCK]")
        if isinstance(content, dict):
            lines.append("```json")
            lines.append(json.dumps(content, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            lines.append(str(content))
        lines.append("")


def build_file_summary(page_summaries: list[str], file_name: str) -> str:
    """Combine per-page summaries into a full file-level summary (no LLM, stored in DB).

    Uses all page summaries for completeness.
    """
    snippet = "\n".join(
        f"- Page {i+1}: {s}" for i, s in enumerate(page_summaries)
    )
    return (
        f"File: {file_name}\n"
        f"Total pages: {len(page_summaries)}\n\n"
        f"Pages overview:\n{snippet}"
    )


def generate_file_short_summary(file_name: str, page_summaries: list[str]) -> str:
    """Use Gemini Flash to generate a concise 2-3 sentence summary of the file.

    This short summary is used inside the folder-level summary so agents can
    quickly understand what each file is about without reading all page details.
    Falls back to a simple concatenation if the API call fails.
    """
    from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
    from google import genai  # type: ignore

    snippet = "\n".join(
        f"- Page {i+1}: {s}" for i, s in enumerate(page_summaries[:10])
    )
    prompt = (
        f"You are summarizing a CAD/architectural drawing file.\n\n"
        f"File name: {file_name}\n"
        f"Total pages: {len(page_summaries)}\n\n"
        f"Per-page summaries:\n{snippet}\n\n"
        f"Write a concise 2-3 sentence summary in Japanese that describes what this file "
        f"contains (e.g. project name, building, types of drawings). "
        f"Be specific and factual. No bullet points, plain prose only."
    )
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        return page_summaries[0] if page_summaries else file_name


def build_folder_summary(file_short_summaries: list[str]) -> str:
    """Aggregate file short summaries into a folder-level summary."""
    total = len(file_short_summaries)
    combined = "\n\n".join(
        f"[File {i+1}]\n{s}" for i, s in enumerate(file_short_summaries[:20])
    )
    return f"Folder contains {total} file(s).\n\n{combined}"
