"""page_agent.py — Two-stage page-level reasoning agent.

Stage 1 — Page Selector (Gemini 2.5 Flash):
  - Input: query + short_summary of ≤25 pages
  - Output: top 3–5 most relevant page numbers

Stage 2 — Page Reasoner (Gemini 3.1 Pro):
  - Input: query + FULL context_md of selected pages (no truncation)
  - Output: detailed answer + pages_used + need_tool

Tool routing (via agents/tool_router.py):
  Tool decision is finalized after Stage 2 (with page context available),
  then dispatched to:
    search       → search_tool.run_search_tool
    count        → count_tool.run_count_tool  (+ Gemini Vision if image)
    area         → area_tool.run_area_tool    (+ Gemini Vision if image)
    viz          → viz_tool (existing)
    report_pdf   → report_tool.run_report_pdf
    report_excel → report_tool.run_report_excel
    none         → plain Q&A answer
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from collections.abc import Callable

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, AGENT_MAX_PAGES
from cad_pipeline.agents.language_utils import detect_query_language, language_label

MAX_SELECTED_PAGES = 5


def _build_citations_from_pages(result_pages_used: list[int], selected_pages: list[dict]) -> list[dict]:
    citations: list[dict] = []
    seen: set[tuple[str, int]] = set()
    page_set = set(result_pages_used or [])
    scoped = [p for p in selected_pages if p.get("page_number") in page_set] or selected_pages
    for p in scoped:
        fid = str(p.get("file_id", ""))
        pno = int(p.get("page_number", 0) or 0)
        if not fid or pno <= 0:
            continue
        key = (fid, pno)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "file_id": fid,
                "file_name": str(p.get("file_name") or fid),
                "page_number": pno,
            }
        )
    return citations


def _parse_reasoner_json(raw: str) -> dict:
    """Parse reasoner JSON robustly; tolerate wrapped text/fences/partial noise."""
    text = (raw or "").strip()
    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    if not text:
        raise ValueError("Empty model response")

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try extracting first JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Last-resort field extraction (especially for malformed JSON with valid quoted answer)
    answer_match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.S)
    pages_match = re.search(r'"pages_used"\s*:\s*\[([^\]]*)\]', text, flags=re.S)
    images_match = re.search(r'"images"\s*:\s*\[([^\]]*)\]', text, flags=re.S)
    tool_match = re.search(r'"need_tool"\s*:\s*"(none|search|count|area|viz|report_pdf|report_excel)"', text)

    if answer_match:
        answer_raw = answer_match.group(1)
        # Use JSON string decoding to preserve UTF-8 text correctly.
        # `unicode_escape` can corrupt non-ASCII characters (mojibake).
        answer = json.loads(f'"{answer_raw}"')
        pages_used: list[int] = []
        if pages_match:
            pages_used = [int(x) for x in re.findall(r"\d+", pages_match.group(1))]
        images: list[str] = []
        if images_match:
            images = re.findall(r'"([^"]+)"', f'[{images_match.group(1)}]')
        need_tool = tool_match.group(1) if tool_match else "none"
        return {
            "answer": answer,
            "pages_used": pages_used,
            "images": images,
            "need_tool": need_tool,
        }

    raise ValueError("Invalid JSON from model")


def _select_pages(
    query: str,
    pages: list[dict],
    client,
    chat_history: list[dict] | None,
) -> list[dict]:
    """Stage 1: Flash selects up to MAX_SELECTED_PAGES most relevant pages."""
    history_text = ""
    if chat_history:
        lines = []
        for turn in chat_history[-5:]:
            lines.append(f"User: {turn['role_user']}")
            lines.append(f"Assistant: {turn['role_assistant']}")
        history_text = "\nPrevious conversation:\n" + "\n".join(lines) + "\n"

    pages_index = "\n".join(
        f"- page={p['page_number']}: {p.get('short_summary', '(no summary)')}"
        for p in pages
    )

    prompt = f"""You are selecting the most relevant pages from a CAD document to answer a question.
{history_text}
Available pages:
{pages_index}

User question: "{query}"

    Select up to {MAX_SELECTED_PAGES} page numbers (ideally 1–3, max 5) most likely to contain the answer.
- Prefer pages whose summary mentions the topic directly
- For count/area questions, include pages with tables or diagrams
- If unsure, include more rather than fewer

Reply ONLY as JSON: {{"page_numbers": [<int>, ...]}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        selected_nums = set(result.get("page_numbers", []))
    except Exception:
        # Fallback: first MAX_SELECTED_PAGES pages
        selected_nums = {p["page_number"] for p in pages[:MAX_SELECTED_PAGES]}

    selected = [p for p in pages if p["page_number"] in selected_nums]
    # Always return at least 1 page
    return selected or pages[:1]


def run_page_agent(
    query: str,
    pages: list[dict],
    use_tools: bool = True,
    chat_history: list[dict] | None = None,
    folder_id: str | None = None,
    file_id: str | None = None,
    image_bytes: bytes | None = None,
    answer_stream_callback: Callable[[str], None] | None = None,
) -> dict:
    """Two-stage page-level reasoning agent.

    Args:
        query:        User's question.
        pages:        List of page dicts {page_id, page_number, context_md,
                      image_url, short_summary}.
        use_tools:    Whether to automatically invoke count/area/search tools.
        chat_history: Last N chat turns for context continuity.
        folder_id:    Folder scope — passed to search_tool when triggered.
        file_id:      File scope — passed to search_tool when triggered.
        image_bytes:  Optional image bytes uploaded by user — passed to
                      search_tool for image-based semantic search.

    Returns:
        {"answer", "pages_used", "images", "tool_result", "need_tool"}
    """
    from google import genai  # type: ignore
    from cad_pipeline.agents.tool_router import classify_tool

    client = genai.Client(api_key=GEMINI_API_KEY)
    pages = pages[:AGENT_MAX_PAGES]
    lang_code = detect_query_language(query)
    lang = language_label(lang_code)

    # ── Stage 1: Flash selects top pages ────────────────────────────────────
    selected_pages = _select_pages(query, pages, client, chat_history)

    tool_result: dict | None = None

    # ── Stage 2: Pro reasons over full context ──────────────────────────────
    history_text = ""
    if chat_history:
        turns = []
        for turn in chat_history[-5:]:
            turns.append(f"User: {turn['role_user']}")
            turns.append(f"Assistant: {turn['role_assistant']}")
        history_text = "\nPrevious conversation:\n" + "\n".join(turns) + "\n"

    pages_text = ""
    for p in selected_pages:
        pages_text += (
            f"\n\n=== PAGE {p['page_number']} ===\n"
            f"Summary: {p.get('short_summary', '')}\n"
            f"Image: {p.get('image_url', '')}\n"
            f"Content:\n{p.get('context_md', '')}"
        )

    prompt = f"""You are a Page-level assistant for a CAD architectural drawing Q&A system.
{history_text}
You have access to the following document pages (full content):
{pages_text}

User question: "{query}"
Detected user language: {lang}

Tasks:
1. Answer accurately using the page content above.
2. Cite the page numbers used.
3. Say clearly if the content doesn't contain the answer.
4. Suggest whether a specialist tool is needed.

Do NOT hallucinate content not in the pages.
The "answer" text MUST be in {lang} only.

Reply ONLY as JSON:
{{
  "answer": "<detailed answer in the same language as the question>",
  "pages_used": [<page_number>, ...],
  "images": ["<image_url if relevant>", ...],
  "need_tool": "none" | "search" | "count" | "area" | "viz" | "report_pdf" | "report_excel"
}}"""

    try:
        if answer_stream_callback is not None:
            # Native Gemini streaming; extract only answer text chunks from streamed JSON.
            raw_parts: list[str] = []
            seen_answer = False
            in_answer = False
            escaped = False
            stream_buf = ""

            for chunk in client.models.generate_content_stream(model=GEMINI_PRO_MODEL, contents=prompt):
                text = getattr(chunk, "text", "") or ""
                if not text:
                    continue
                raw_parts.append(text)
                stream_buf += text

                if not seen_answer:
                    marker_idx = stream_buf.find('"answer"')
                    if marker_idx >= 0:
                        colon_idx = stream_buf.find(":", marker_idx)
                        first_quote_idx = stream_buf.find('"', colon_idx + 1 if colon_idx >= 0 else marker_idx + 8)
                        if first_quote_idx >= 0:
                            seen_answer = True
                            in_answer = True
                            escaped = False
                            stream_buf = stream_buf[first_quote_idx + 1 :]
                        else:
                            # keep tail in case marker split across chunks
                            stream_buf = stream_buf[max(0, len(stream_buf) - 64):]
                            continue
                    else:
                        stream_buf = stream_buf[max(0, len(stream_buf) - 64):]
                        continue

                if in_answer:
                    out_chars: list[str] = []
                    i = 0
                    while i < len(stream_buf):
                        ch = stream_buf[i]
                        if escaped:
                            # minimal unescape for common JSON sequences
                            if ch == "n":
                                out_chars.append("\n")
                            elif ch == "t":
                                out_chars.append("\t")
                            elif ch in {'"', "\\", "/"}:
                                out_chars.append(ch)
                            else:
                                out_chars.append(ch)
                            escaped = False
                            i += 1
                            continue
                        if ch == "\\":
                            escaped = True
                            i += 1
                            continue
                        if ch == '"':
                            in_answer = False
                            i += 1
                            break
                        out_chars.append(ch)
                        i += 1

                    if out_chars:
                        answer_stream_callback("".join(out_chars))
                    stream_buf = stream_buf[i:]
                    if not in_answer:
                        stream_buf = ""
                        break

            raw = "".join(raw_parts).strip()
        else:
            response = client.models.generate_content(model=GEMINI_PRO_MODEL, contents=prompt)
            raw = response.text.strip()
        result = _parse_reasoner_json(raw)
    except Exception as exc:
        result = {"answer": f"Error during page reasoning: {exc}",
                  "pages_used": [], "images": [], "need_tool": "none"}

    # Final tool decision at the end of pipeline:
    # - Gemini tool_router provides intent from user input (query/image)
    # - Stage-2 reasoner provides context-aware suggestion
    # Router wins when explicit, else use reasoner suggestion.
    routed_tool = classify_tool(query=query, image_bytes=image_bytes).get("tool", "none") if use_tools else "none"
    suggested_tool = result.get("need_tool", "none")
    need_tool = routed_tool if routed_tool != "none" else suggested_tool

    if not use_tools or need_tool == "none":
        result["tool_result"] = None
        result["selected_pages"] = [p["page_number"] for p in selected_pages]
        result["citations"] = _build_citations_from_pages(
            result.get("pages_used", []),
            selected_pages,
        )
        return result

    # ── Dispatch tools that benefit from Stage 2 context ───────────────────
    used_nums  = set(result.get("pages_used", []))
    used_pages = [p for p in selected_pages if p["page_number"] in used_nums] or selected_pages
    combined_ctx = "\n\n".join(p.get("context_md", "") for p in used_pages)

    if need_tool == "search":
        from cad_pipeline.tools.search_tool import run_search_tool
        tool_result = run_search_tool(
            query=query, image_bytes=image_bytes,
            folder_id=folder_id, file_id=file_id, top_n=10,
        )
        hits = tool_result.get("results", [])
        if hits:
            answer_lines = []
            if tool_result.get("image_description"):
                answer_lines.append(
                    f"*{'Mô tả ảnh' if lang_code == 'vi' else '画像説明' if lang_code == 'ja' else 'Image description'}:* "
                    f"{tool_result['image_description']}\n"
                )
            answer_lines.append(
                f"**{'Top ' + str(len(hits)) + ' trang liên quan' if lang_code == 'vi' else '関連ページ上位 ' + str(len(hits)) + ' 件' if lang_code == 'ja' else 'Top ' + str(len(hits)) + ' relevant pages'}:**"
            )
            for h in hits:
                page_word = "trang" if lang_code == "vi" else "ページ" if lang_code == "ja" else "page"
                answer_lines.append(f"- **{h['file_name']}** {page_word} {h['page_number']} (score {h['vector_score']:.3f}): {h['short_summary'][:120]}")
            result["answer"] = "\n".join(answer_lines)
        else:
            result["answer"] = (
                "Không tìm thấy trang liên quan."
                if lang_code == "vi"
                else "関連ページが見つかりませんでした。"
                if lang_code == "ja"
                else "No relevant pages found."
            )

    if need_tool == "count":
        from cad_pipeline.tools.count_tool import run_count_tool
        # Save image to temp file so count_tool Vision mode can use it
        _img_path: str | None = None
        if image_bytes:
            _tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            _tmp.write(image_bytes); _tmp.close()
            _img_path = _tmp.name
        # If user uploaded an image, prioritize counting on that image instead of DXF/page context.
        dxf_path = None if _img_path else next((p.get("dxf_path") for p in used_pages if p.get("dxf_path")), None)
        tool_result = run_count_tool(
            query=query, dxf_path=dxf_path, context_md=combined_ctx,
            image_path=_img_path,
        )
        count_value = tool_result.get("count")
        if count_value is not None:
            label = "Kết quả đếm" if lang_code == "vi" else "カウント結果" if lang_code == "ja" else "Count result"
            result["answer"] += f"\n\n**{label}:** {count_value} — {tool_result.get('details', '')}"

    elif need_tool == "area":
        from cad_pipeline.tools.area_tool import run_area_tool
        _img_path = None
        if image_bytes:
            _tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            _tmp.write(image_bytes); _tmp.close()
            _img_path = _tmp.name
        tool_result = run_area_tool(
            query=query, image_path=_img_path, context_md=combined_ctx,
        )
        area_value = tool_result.get("area")
        if area_value is None:
            area_value = tool_result.get("total_m2")
        if area_value is not None:
            label = "Kết quả diện tích" if lang_code == "vi" else "面積結果" if lang_code == "ja" else "Area result"
            result["answer"] += f"\n\n**{label}:** {area_value} {tool_result.get('unit', 'm²')} — {tool_result.get('details', '')}"

    elif need_tool == "report_pdf":
        from cad_pipeline.tools.report_tool import run_report_pdf
        _pages_enriched = [
            {**p, "file_name": p.get("file_name", p.get("file_id", ""))}
            for p in used_pages
        ]
        tool_result = run_report_pdf(query=query, pages=_pages_enriched, tool_result=None)
        if tool_result.get("success"):
            result["answer"] = (
                f"Đã tạo báo cáo PDF: **{tool_result['file_name']}**"
                if lang_code == "vi"
                else f"PDFレポートを作成しました: **{tool_result['file_name']}**"
                if lang_code == "ja"
                else f"Generated PDF report: **{tool_result['file_name']}**"
            )
        else:
            result["answer"] = (
                f"Lỗi tạo PDF: {tool_result.get('error')}"
                if lang_code == "vi"
                else f"PDF作成エラー: {tool_result.get('error')}"
                if lang_code == "ja"
                else f"PDF generation error: {tool_result.get('error')}"
            )

    elif need_tool == "report_excel":
        from cad_pipeline.tools.report_tool import run_report_excel
        _pages_enriched = [
            {**p, "file_name": p.get("file_name", p.get("file_id", ""))}
            for p in used_pages
        ]
        tool_result = run_report_excel(
            query=query,
            pages=_pages_enriched,
            tool_result=tool_result,
            answer_text=result.get("answer", ""),
            chat_history=chat_history,
        )
        if tool_result.get("success"):
            result["answer"] = (
                f"Đã tạo file Excel: **{tool_result['file_name']}**"
                if lang_code == "vi"
                else f"Excelファイルを作成しました: **{tool_result['file_name']}**"
                if lang_code == "ja"
                else f"Generated Excel file: **{tool_result['file_name']}**"
            )
        else:
            result["answer"] = (
                f"Lỗi tạo Excel: {tool_result.get('error')}"
                if lang_code == "vi"
                else f"Excel作成エラー: {tool_result.get('error')}"
                if lang_code == "ja"
                else f"Excel generation error: {tool_result.get('error')}"
            )

    result["tool_result"] = tool_result
    result["selected_pages"] = [p["page_number"] for p in selected_pages]
    result["citations"] = _build_citations_from_pages(
        result.get("pages_used", []),
        selected_pages,
    )
    return result
