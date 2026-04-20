"""qa_pipeline.py — Full Q&A pipeline: Router → Folder → File → Page → Tool.

Usage:
  from cad_pipeline.pipeline.qa_pipeline import run_qa
  result = run_qa(query="バルブは何個ありますか？", folder_id="folder_001")
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from collections.abc import Callable

from cad_pipeline.agents.router import classify_query
from cad_pipeline.agents.tool_router import classify_tool
from cad_pipeline.agents.language_utils import detect_query_language
from cad_pipeline.agents.folder_agent import run_folder_agent
from cad_pipeline.agents.file_agent import run_file_agent
from cad_pipeline.agents.page_agent import run_page_agent
from cad_pipeline.observability.phoenix_tracing import traced_span
from cad_pipeline.storage import mongo


def run_qa(
    query: str,
    folder_id: str,
    file_id: str | None = None,
    session_file_ids: list[str] | None = None,
    user_email: str | None = None,
    save_history: bool = True,
    image_bytes: bytes | None = None,
    user_image_url: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    answer_stream_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run the full Q&A pipeline for a given query.

    Args:
        query:        User's natural language question.
        folder_id:    Target folder to search within.
        file_id:      Optional file to narrow search scope.
        save_history: Persist this turn to MongoDB chat_history.
        image_bytes:  Optional image bytes uploaded with the query.
                      Triggers image-based semantic search via search_tool
                      when the query intent is "search" or image alone is
                      provided.

    Returns:
        {
          "answer": str,
          "pages_used": [int],
          "images": [str],
          "tool_result": dict | None,
          "source_file": str,
          "query_type": "qa",
        }
    """
    def _emit_event(event: dict) -> None:
        if progress_callback is None:
            return
        payload = {
            "phase": str(event.get("phase", "step")),
            "message": str(event.get("message", "")),
            "step_id": str(event.get("step_id", "")),
            "tool": str(event.get("tool", "")),
            "status": str(event.get("status", "")),
        }
        progress_callback(payload)

    def _emit(step: str) -> None:
        _emit_event({"phase": "step", "message": step, "status": "running"})

    def _trim(value: str | None, limit: int = 240) -> str:
        if not value:
            return ""
        text = value.strip().replace("\n", " ")
        return text[:limit]

    def _set_attr(span: object | None, key: str, value: object) -> None:
        if span is None or value is None:
            return
        try:
            span.set_attribute(key, value)  # type: ignore[attr-defined]
        except Exception:
            try:
                span.set_attribute(key, str(value))  # type: ignore[attr-defined]
            except Exception:
                pass

    def _add_event(span: object | None, name: str, attrs: dict[str, object] | None = None) -> None:
        if span is None:
            return
        try:
            span.add_event(name, attrs or {})  # type: ignore[attr-defined]
        except Exception:
            pass

    def _sanitize_ui_updates(raw: object) -> list[dict]:
        if not isinstance(raw, list):
            return []
        updates: list[dict] = []
        for item in raw[:6]:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message", "")).strip()
            if not message:
                continue
            updates.append(
                {
                    "phase": str(item.get("phase", "step_started")).strip() or "step_started",
                    "message": message,
                    "step_id": str(item.get("step_id", "")).strip(),
                    "tool": str(item.get("tool", "")).strip(),
                    "status": str(item.get("status", "running")).strip() or "running",
                }
            )
        return updates

    def _emit_ui_updates(raw: object) -> None:
        for upd in _sanitize_ui_updates(raw):
            _emit_event(upd)

    lang = detect_query_language(query)

    def _msg(vi: str, ja: str, en: str) -> str:
        return vi if lang == "vi" else ja if lang == "ja" else en

    def _history_context(turns: list[dict], limit: int = 5) -> str:
        lines: list[str] = []
        for t in turns[-limit:]:
            lines.append(f"User: {t.get('role_user', '')}")
            lines.append(f"Assistant: {t.get('role_assistant', '')}")
        return "\n".join(lines)

    def _save_turn(
        answer_text: str,
        *,
        citations: list[dict] | None = None,
        images: list[str] | None = None,
        tool_result: dict | None = None,
        source_file: str | None = None,
    ) -> None:
        if not save_history or not answer_text:
            return
        assistant_image_url = ""
        if isinstance(tool_result, dict):
            assistant_image_url = str(tool_result.get("image_url") or "").strip()
        if not assistant_image_url and images:
            assistant_image_url = str(images[0] or "").strip()
        download_url = ""
        if isinstance(tool_result, dict):
            explicit_url = str(tool_result.get("download_url") or "").strip()
            if explicit_url:
                download_url = explicit_url
            else:
                report_format = str(tool_result.get("format") or "").strip().lower()
                report_path = str(tool_result.get("file_path") or "").strip()
                if report_format in {"pdf", "excel"} and report_path:
                    report_name = Path(report_path).name
                    if report_name:
                        download_url = f"/reports/{report_name}"
        user_meta = {"image_url": user_image_url} if user_image_url else {}
        assistant_meta = {
            "citations": citations or [],
            "image_url": assistant_image_url,
            "download_url": download_url,
            "source_file": source_file or "",
        }
        mongo.append_chat_turn(
            folder_id,
            query,
            answer_text,
            user_email=user_email,
            user_meta=user_meta,
            assistant_meta=assistant_meta,
        )

    def _norm_text(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"\s+", " ", text)

    def _extract_title_block_from_image(query_text: str, img_bytes: bytes) -> dict[str, str]:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np
            from google import genai  # type: ignore
            from google.genai import types as _gt  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
            from cad_pipeline.core.layout_detect import LayoutDetector

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""Extract title-block metadata from this architectural drawing image.

User query: "{query_text}"

Return ONLY JSON:
{{
  "drawing_no": "<string or empty>",
  "drawing_title": "<string or empty>",
  "project": "<string or empty>"
}}"""
            def _extract_from_bytes(source_bytes: bytes) -> dict[str, str]:
                resp = client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=[
                        _gt.Part.from_bytes(data=source_bytes, mime_type="image/png"),
                        prompt,
                    ],
                )
                raw = str(getattr(resp, "text", "") or "").strip()
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                return {
                    "drawing_no": str(parsed.get("drawing_no", "")).strip(),
                    "drawing_title": str(parsed.get("drawing_title", "")).strip(),
                    "project": str(parsed.get("project", "")).strip(),
                }

            # Prefer cropped title-block region from layout detector.
            try:
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is not None:
                    detector = LayoutDetector.get()
                    blocks = detector.predict_image(image)
                    title_blocks = [b for b in blocks if b.label == "title_block" and b.width > 0 and b.height > 0]
                    if title_blocks:
                        best = max(title_blocks, key=lambda b: float(b.score) * max(1, b.width * b.height))
                        crop = best.crop(image)
                        ok, enc = cv2.imencode(".png", crop)
                        if ok:
                            return _extract_from_bytes(enc.tobytes())
            except Exception:
                pass

            # Fallback to full image when no reliable title block crop exists.
            return _extract_from_bytes(img_bytes)
        except Exception:
            return {"drawing_no": "", "drawing_title": "", "project": ""}

    def _best_title_block_match(title_meta: dict[str, str], scoped_files: list[dict]) -> dict | None:
        q_no = _norm_text(title_meta.get("drawing_no"))
        q_title = _norm_text(title_meta.get("drawing_title"))
        q_project = _norm_text(title_meta.get("project"))
        if not (q_no or q_title or q_project):
            return None

        best: dict | None = None
        best_score = 0.0
        for fdoc in scoped_files:
            rows = fdoc.get("title_block_index") or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                score = 0.0
                no = _norm_text(row.get("drawing_no"))
                title = _norm_text(row.get("drawing_title"))
                project = _norm_text(row.get("project"))

                if q_no and no:
                    if q_no == no:
                        score += 0.75
                    elif q_no in no or no in q_no:
                        score += 0.45
                if q_title and title:
                    if q_title == title:
                        score += 0.5
                    elif q_title in title or title in q_title:
                        score += 0.3
                if q_project and project:
                    if q_project == project:
                        score += 0.2
                    elif q_project in project or project in q_project:
                        score += 0.1

                if score > best_score:
                    best_score = score
                    best = {
                        "file_id": str(fdoc.get("_id", "")),
                        "file_name": str(fdoc.get("file_name", "") or fdoc.get("_id", "")),
                        "page_number": int(row.get("page_number") or 0),
                        "score": score,
                        "row": row,
                    }
        if best is None:
            return None
        if best_score < 0.45:
            return None
        return best

    def _try_direct_non_doc_answer(query_text: str) -> dict[str, object] | None:
        """Directly answer casual/off-topic chat without running document pipeline."""
        if not query_text.strip():
            return None
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""You are a chat assistant in a CAD-document Q&A app.

Task:
Decide whether the user query can be answered directly WITHOUT using any document/image/page context.

Use direct answer only for:
- greetings, casual small talk
- general chit-chat
- broad non-document questions unrelated to CAD files, pages, uploaded images, architectural plan analysis

Do NOT use direct answer for:
- questions that depend on project documents/pages/files/citations
- uploaded image analysis
- CAD/layout count/area/search/report tasks

User query:
"{query_text}"

Reply ONLY JSON:
{{
  "is_direct": true | false,
  "answer": "<direct helpful answer in user's language, empty if is_direct=false>",
  "reason": "<short reason>"
}}"""
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            raw = str(getattr(resp, "text", "") or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            is_direct = bool(parsed.get("is_direct", False))
            if not is_direct:
                return None
            answer = str(parsed.get("answer", "")).strip()
            if not answer:
                return None
            return {
                "answer": answer,
                "reason": _trim(str(parsed.get("reason", "")), 160) or "direct_non_doc",
            }
        except Exception:
            return None

    def _history_debug(turns: list[dict], limit: int = 5) -> dict[str, object]:
        scoped = turns[-limit:]
        payload: dict[str, object] = {"history_turns": len(scoped)}
        for i, turn in enumerate(scoped, start=1):
            payload[f"turn_{i}_user"] = _trim(str(turn.get("role_user", "")), 180)
            payload[f"turn_{i}_assistant"] = _trim(str(turn.get("role_assistant", "")), 180)
        return payload

    def _run_tool_from_history(
        tool_name: str,
        query_text: str,
        turns: list[dict],
    ) -> dict | None:
        """Try tool execution directly from chat history context.

        Returns a full run_qa-compatible response dict on success, else None.
        """
        if not turns:
            return None

        context_text = _history_context(turns)
        if not context_text.strip():
            return None

        if tool_name == "count":
            from cad_pipeline.tools.count_tool import run_count_tool

            tool_result = run_count_tool(query=query_text, context_md=context_text)
            count_value = tool_result.get("count")
            confidence = str(tool_result.get("confidence", "")).lower()
            if count_value is None:
                return None
            try:
                count_num = int(count_value)
            except Exception:
                count_num = 0
            # History shortcut should only short-circuit when we have a positive,
            # usable count. Zero/uncertain values must continue to page-level lookup.
            if count_num <= 0:
                return None
            if confidence == "low":
                return None
            answer = str(tool_result.get("details") or "").strip() or str(count_value)
            return {
                "answer": answer,
                "pages_used": [],
                "citations": [],
                "images": [],
                "tool_result": tool_result,
                "source_file": "chat_history",
                "query_type": "qa",
            }

        if tool_name == "area":
            from cad_pipeline.tools.area_tool import run_area_tool

            tool_result = run_area_tool(query=query_text, context_md=context_text)
            area_value = tool_result.get("area")
            if area_value in (None, "", "unknown"):
                area_value = tool_result.get("total_m2")
            if area_value in (None, "", "unknown"):
                return None
            answer = str(tool_result.get("details") or "").strip() or str(area_value)
            return {
                "answer": answer,
                "pages_used": [],
                "citations": [],
                "images": [],
                "tool_result": tool_result,
                "source_file": "chat_history",
                "query_type": "qa",
            }

        pages_like = [
            {
                "page_number": 0,
                "file_name": "chat_history",
                "short_summary": "Conversation-derived context",
                "context_md": context_text,
            }
        ]

        if tool_name == "report_pdf":
            from cad_pipeline.tools.report_tool import run_report_pdf

            tool_result = run_report_pdf(query=query_text, pages=pages_like, tool_result=None)
            if not tool_result.get("file_path"):
                return None
            answer = _msg(
                f"Đã tạo báo cáo PDF từ ngữ cảnh hội thoại: **{tool_result.get('file_name', '')}**",
                f"会話コンテキストからPDFレポートを作成しました: **{tool_result.get('file_name', '')}**",
                f"Generated PDF report from chat context: **{tool_result.get('file_name', '')}**",
            )
            return {
                "answer": answer,
                "pages_used": [],
                "citations": [],
                "images": [],
                "tool_result": tool_result,
                "source_file": "chat_history",
                "query_type": "qa",
            }

        if tool_name == "report_excel":
            from cad_pipeline.tools.report_tool import run_report_excel

            tool_result = run_report_excel(
                query=query_text,
                pages=pages_like,
                tool_result=None,
                answer_text=(turns[-1].get("role_assistant", "") if turns else ""),
                chat_history=turns,
            )
            if not tool_result.get("file_path"):
                return None
            answer = _msg(
                f"Đã tạo file Excel từ ngữ cảnh hội thoại: **{tool_result.get('file_name', '')}**",
                f"会話コンテキストからExcelファイルを作成しました: **{tool_result.get('file_name', '')}**",
                f"Generated Excel file from chat context: **{tool_result.get('file_name', '')}**",
            )
            return {
                "answer": answer,
                "pages_used": [],
                "citations": [],
                "images": [],
                "tool_result": tool_result,
                "source_file": "chat_history",
                "query_type": "qa",
            }

        return None

    def _run_orchestrator_plan(
        query_text: str,
        turns: list[dict],
        has_image: bool,
        routed_tool_hint: str,
    ) -> dict[str, object]:
        """Plan execution with a single Pro-model orchestrator."""
        tool_options = {"none", "search", "count", "area", "viz", "report_pdf", "report_excel"}
        default_plan: dict[str, object] = {
            "grounding_mode": "uploaded_image" if has_image else "page_context",
            "preferred_tool": routed_tool_hint if routed_tool_hint in tool_options else "none",
            "use_history_shortcut": not has_image,
            "force_page_level": routed_tool_hint in {"count", "area", "viz", "search", "report_pdf", "report_excel"},
            "allow_folder_direct_answer": routed_tool_hint == "none",
            "allow_file_direct_answer": routed_tool_hint == "none",
            "allow_image_early_answer": has_image,
            "early_exit_confidence": "medium",
            "reason": "default_fallback_plan",
            "ui_updates": [],
        }
        history_text = _history_context(turns, limit=6)

        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_PRO_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""You are the Orchestrator Agent for a CAD assistant pipeline.

Your job is to decide a smooth execution plan with early-exit when enough evidence is available.

Available specialized tools/agents:
- history_tool: reuse recent chat history for count/area/report shortcuts
- folder_agent: choose relevant files or answer only very high-level overview
- file_agent: decide direct summary answer vs candidate pages
- page_agent: full page-level reasoning + downstream tool execution
- image_tools: direct uploaded-image execution for count/area/general vision QA

Inputs:
- has_uploaded_image: {str(has_image).lower()}
- tool_router_hint: "{routed_tool_hint}"
- user_query: "{query_text}"
- recent_conversation:
---
{history_text}
---

Return ONLY JSON:
{{
  "grounding_mode": "uploaded_image" | "page_context" | "hybrid",
  "preferred_tool": "none" | "search" | "count" | "area" | "viz" | "report_pdf" | "report_excel",
  "use_history_shortcut": true | false,
  "force_page_level": true | false,
  "allow_folder_direct_answer": true | false,
  "allow_file_direct_answer": true | false,
  "allow_image_early_answer": true | false,
  "early_exit_confidence": "high" | "medium" | "low",
  "reason": "<short rationale>",
  "ui_updates": [
    {{
      "phase": "plan_created" | "step_started" | "tool_called" | "tool_result" | "replan" | "finalizing",
      "message": "<natural short update in user's language, context-aware>",
      "step_id": "<optional>",
      "tool": "<optional>",
      "status": "running" | "done" | "error"
    }}
  ]
}}

Rules:
- Avoid rigid flow; allow early exit when confidence is sufficient.
- If user intent is clearly about uploaded image details, grounding_mode should favor uploaded_image.
- If user intent shifts back to document/page citations, grounding_mode should favor page_context.
- If uncertain, choose hybrid.
- Keep plan conservative: avoid hallucination and preserve citation quality.
"""
            resp = client.models.generate_content(
                model=GEMINI_PRO_MODEL,
                contents=prompt,
            )
            raw = str(getattr(resp, "text", "") or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            plan = dict(default_plan)

            grounding = str(parsed.get("grounding_mode", plan["grounding_mode"])).lower()
            if grounding not in {"uploaded_image", "page_context", "hybrid"}:
                grounding = str(plan["grounding_mode"])
            preferred_tool = str(parsed.get("preferred_tool", plan["preferred_tool"])).lower()
            if preferred_tool not in tool_options:
                preferred_tool = str(plan["preferred_tool"])
            early_conf = str(parsed.get("early_exit_confidence", "medium")).lower()
            if early_conf not in {"high", "medium", "low"}:
                early_conf = "medium"

            plan.update(
                {
                    "grounding_mode": grounding,
                    "preferred_tool": preferred_tool,
                    "use_history_shortcut": bool(parsed.get("use_history_shortcut", plan["use_history_shortcut"])),
                    "force_page_level": bool(parsed.get("force_page_level", plan["force_page_level"])),
                    "allow_folder_direct_answer": bool(parsed.get("allow_folder_direct_answer", plan["allow_folder_direct_answer"])),
                    "allow_file_direct_answer": bool(parsed.get("allow_file_direct_answer", plan["allow_file_direct_answer"])),
                    "allow_image_early_answer": bool(parsed.get("allow_image_early_answer", plan["allow_image_early_answer"])),
                    "early_exit_confidence": early_conf,
                    "reason": _trim(str(parsed.get("reason", "")), 180) or "model_plan",
                    "ui_updates": _sanitize_ui_updates(parsed.get("ui_updates")),
                }
            )
            return plan
        except Exception as exc:
            default_plan["reason"] = f"fallback_on_error:{_trim(str(exc), 80)}"
            return default_plan

    def _orchestrator_replan_after_step(
        query_text: str,
        plan: dict[str, object],
        step_name: str,
        provisional_answer: str,
        tool_result: dict | None = None,
        pages_used_count: int = 0,
        citations_count: int = 0,
    ) -> dict[str, object]:
        """Ask orchestrator whether to finalize now or continue."""
        allowed_steps = {
            "history_shortcut",
            "image_count",
            "image_area",
            "image_general_qa",
            "folder_agent",
            "file_agent",
            "page_agent",
            "finalize",
        }
        answer = provisional_answer.strip()
        if not answer:
            return {
                "finalize_now": False,
                "next_step": "continue_page_level",
                "next_tool_sequence": ["page_agent"],
                "reason": "empty_answer",
                "ui_updates": [],
            }

        confidence_hint = str((tool_result or {}).get("confidence", "")).lower() if tool_result else ""
        early_exit_target = str(plan.get("early_exit_confidence", "medium")).lower()
        default_finalize = True
        if confidence_hint == "low":
            default_finalize = False
        if early_exit_target == "high" and confidence_hint not in {"high"} and citations_count == 0 and step_name.startswith("image_first"):
            default_finalize = False
        default_sequence = ["page_agent"] if not default_finalize else ["finalize"]

        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_PRO_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""You are the Orchestrator Agent deciding whether to stop early.

User query: "{query_text}"
Current step: "{step_name}"
Current plan (JSON): {json.dumps(plan, ensure_ascii=False)}
Provisional answer:
---
{answer[:1600]}
---
Tool result summary (JSON): {json.dumps(tool_result or {}, ensure_ascii=False)[:1600]}
Evidence stats: pages_used={pages_used_count}, citations={citations_count}, confidence_hint="{confidence_hint or "unknown"}"

Decide if we should finalize now or continue pipeline for better grounding.
If continuing, provide explicit next_tool_sequence from allowed steps.

Return ONLY JSON:
{{
  "finalize_now": true | false,
  "next_step": "finalize" | "continue_page_level",
  "next_tool_sequence": ["history_shortcut" | "image_count" | "image_area" | "image_general_qa" | "folder_agent" | "file_agent" | "page_agent" | "finalize", ...],
  "reason": "<short reason>",
  "ui_updates": [
    {{
      "phase": "replan" | "step_started" | "tool_called" | "tool_result" | "finalizing",
      "message": "<natural short update in user's language, context-aware>",
      "step_id": "<optional>",
      "tool": "<optional>",
      "status": "running" | "done" | "error"
    }}
  ]
}}"""
            resp = client.models.generate_content(model=GEMINI_PRO_MODEL, contents=prompt)
            raw = str(getattr(resp, "text", "") or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            finalize_now = bool(parsed.get("finalize_now", default_finalize))
            next_step = str(parsed.get("next_step", "finalize" if finalize_now else "continue_page_level")).lower()
            if next_step not in {"finalize", "continue_page_level"}:
                next_step = "finalize" if finalize_now else "continue_page_level"
            sequence_raw = parsed.get("next_tool_sequence", default_sequence)
            if not isinstance(sequence_raw, list):
                sequence_raw = default_sequence
            sequence = []
            for item in sequence_raw:
                step = str(item).strip().lower()
                if step in allowed_steps and step not in sequence:
                    sequence.append(step)
            if not sequence:
                sequence = list(default_sequence)
            if next_step == "continue_page_level":
                finalize_now = False
                if "page_agent" not in sequence:
                    sequence.append("page_agent")
            else:
                sequence = ["finalize"]
            reason = _trim(str(parsed.get("reason", "")), 180) or "model_replan"
            return {
                "finalize_now": finalize_now,
                "next_step": next_step,
                "next_tool_sequence": sequence,
                "reason": reason,
                "ui_updates": _sanitize_ui_updates(parsed.get("ui_updates")),
            }
        except Exception as exc:
            return {
                "finalize_now": default_finalize,
                "next_step": "finalize" if default_finalize else "continue_page_level",
                "next_tool_sequence": list(default_sequence),
                "reason": f"fallback_on_error:{_trim(str(exc), 80)}",
                "ui_updates": [],
            }

    with traced_span(
        "qa.run",
        query=_trim(query, 300),
        query_length=len(query),
        folder_id=folder_id,
        file_id=file_id or "",
        has_file_id=bool(file_id),
        has_image=bool(image_bytes),
        session_file_count=len(session_file_ids or []),
    ) as root_span:
        _add_event(root_span, "qa.request.received", {"query": _trim(query, 300), "folder_id": folder_id})
        if not image_bytes:
            _emit("Direct-chat route check")
            with traced_span("qa.direct_chat_check", query=_trim(query, 220)) as direct_span:
                direct = _try_direct_non_doc_answer(query)
                if direct is not None:
                    answer = str(direct.get("answer", "")).strip()
                    _set_attr(direct_span, "routed_direct", True)
                    _set_attr(direct_span, "reason", str(direct.get("reason", "")))
                    _set_attr(direct_span, "answer_preview", _trim(answer, 220))
                    _emit_event(
                        {
                            "phase": "finalizing",
                            "message": answer[:160],
                            "status": "done",
                            "step_id": "direct_chat",
                            "tool": "direct_chat",
                        }
                    )
                    _save_turn(answer, source_file="general_chat", tool_result={"mode": "direct_non_doc"})
                    return {
                        "answer": answer,
                        "pages_used": [],
                        "citations": [],
                        "images": [],
                        "tool_result": {"mode": "direct_non_doc"},
                        "source_file": "general_chat",
                        "query_type": "qa",
                    }
                _set_attr(direct_span, "routed_direct", False)
        _emit("Load chat history")
        with traced_span("qa.load_chat_history", folder_id=folder_id, user_email=user_email or "") as history_span:
            chat_history = mongo.get_chat_history(folder_id, user_email=user_email)
            _set_attr(history_span, "chat_history_turns", len(chat_history))
            _set_attr(history_span, "last_user_turn", _trim((chat_history[-1].get("role_user") if chat_history else ""), 180))
            _set_attr(
                history_span,
                "last_assistant_turn",
                _trim((chat_history[-1].get("role_assistant") if chat_history else ""), 180),
            )
            _set_attr(root_span, "chat_history_turns", len(chat_history))
            _add_event(history_span, "qa.chat_history.loaded", _history_debug(chat_history))

        routed_tool_hint = classify_tool(query=query, image_bytes=image_bytes).get("tool", "none")
        if image_bytes and routed_tool_hint in {"none", "search"}:
            _emit("Title-block lookup on uploaded image")
            with traced_span("qa.title_block_lookup", query=_trim(query, 220)) as tb_span:
                if session_file_ids is not None:
                    scoped_files_for_title = mongo.get_files_by_ids(session_file_ids)
                else:
                    scoped_files_for_title = mongo.list_files(folder_id)
                title_meta = _extract_title_block_from_image(query, image_bytes)
                match = _best_title_block_match(title_meta, scoped_files_for_title)
                _set_attr(tb_span, "meta_drawing_no", _trim(title_meta.get("drawing_no", ""), 120))
                _set_attr(tb_span, "meta_drawing_title", _trim(title_meta.get("drawing_title", ""), 140))
                _set_attr(tb_span, "scope_file_count", len(scoped_files_for_title))
                if match is not None:
                    _set_attr(tb_span, "matched_file_id", str(match.get("file_id", "")))
                    _set_attr(tb_span, "matched_file_name", str(match.get("file_name", "")))
                    _set_attr(tb_span, "matched_page", int(match.get("page_number", 0)))
                    _set_attr(tb_span, "matched_score", float(match.get("score", 0.0)))
                    answer = _msg(
                        f"Ảnh bạn gửi khớp với **{match['file_name']}** (trang {match['page_number']}) dựa trên thông tin title block.",
                        f"アップロード画像は title block 情報に基づき **{match['file_name']}**（{match['page_number']}ページ）に一致しました。",
                        f"Your uploaded image matches **{match['file_name']}** (page {match['page_number']}) based on title-block metadata.",
                    )
                    _save_turn(
                        answer,
                        citations=[
                            {
                                "file_id": str(match["file_id"]),
                                "file_name": str(match["file_name"]),
                                "page_number": int(match["page_number"]),
                            }
                        ],
                        source_file=str(match["file_name"]),
                    )
                    return {
                        "answer": answer,
                        "pages_used": [int(match["page_number"])],
                        "citations": [
                            {
                                "file_id": str(match["file_id"]),
                                "file_name": str(match["file_name"]),
                                "page_number": int(match["page_number"]),
                            }
                        ],
                        "images": [],
                        "tool_result": {
                            "mode": "title_block_lookup",
                            "match_score": float(match["score"]),
                            "query_title_block": title_meta,
                            "matched_title_block": match.get("row", {}),
                        },
                        "source_file": str(match["file_name"]),
                        "query_type": "qa",
                    }
                _add_event(
                    tb_span,
                    "qa.title_block_lookup.miss",
                    {"reason": "no_confident_match"},
                )
        with traced_span("qa.orchestrator.plan", query=_trim(query, 220)) as plan_span:
            plan = _run_orchestrator_plan(
                query_text=query,
                turns=chat_history,
                has_image=bool(image_bytes),
                routed_tool_hint=routed_tool_hint,
            )
            _set_attr(plan_span, "grounding_mode", str(plan.get("grounding_mode", "")))
            _set_attr(plan_span, "preferred_tool", str(plan.get("preferred_tool", "")))
            _set_attr(plan_span, "use_history_shortcut", bool(plan.get("use_history_shortcut", False)))
            _set_attr(plan_span, "force_page_level", bool(plan.get("force_page_level", False)))
            _set_attr(plan_span, "allow_image_early_answer", bool(plan.get("allow_image_early_answer", False)))
            _set_attr(plan_span, "reason", str(plan.get("reason", "")))
            _add_event(plan_span, "qa.orchestrator.plan_ready", {k: str(v) for k, v in plan.items()})
            _emit_ui_updates(plan.get("ui_updates"))

        use_uploaded_image = bool(image_bytes) and str(plan.get("grounding_mode", "page_context")) in {"uploaded_image", "hybrid"}
        effective_image_bytes = image_bytes if use_uploaded_image else None
        _set_attr(root_span, "use_uploaded_image", use_uploaded_image)
        _set_attr(root_span, "orchestrator_grounding_mode", str(plan.get("grounding_mode", "")))

        flow_overrides: dict[str, object] = {}

        def _apply_replan_directive(replan: dict[str, object]) -> None:
            sequence = replan.get("next_tool_sequence")
            if not isinstance(sequence, list):
                return
            seq = [str(s).strip().lower() for s in sequence]
            if "page_agent" in seq:
                flow_overrides["force_page_level"] = True
                flow_overrides["allow_folder_direct_answer"] = False
                flow_overrides["allow_file_direct_answer"] = False
            elif "file_agent" in seq:
                flow_overrides["allow_folder_direct_answer"] = False
            elif "folder_agent" in seq:
                flow_overrides["allow_folder_direct_answer"] = True
                flow_overrides["allow_file_direct_answer"] = True

        _emit("Detect query tool intent")
        force_continue_page_level = False
        with traced_span("qa.detect_tool_intent", query=_trim(query, 240)) as tool_span:
            routed_tool = str(plan.get("preferred_tool") or "none")
            if routed_tool not in {"none", "search", "count", "area", "viz", "report_pdf", "report_excel"}:
                routed_tool = classify_tool(query=query, image_bytes=effective_image_bytes).get("tool", "none")
            _set_attr(tool_span, "routed_tool", routed_tool)
            _set_attr(tool_span, "routed_tool_hint", routed_tool_hint)
            allow_history_shortcut = bool(plan.get("use_history_shortcut", False)) and (not effective_image_bytes)
            if allow_history_shortcut and routed_tool in {"count", "area", "report_pdf", "report_excel"}:
                _add_event(
                    tool_span,
                    "qa.tool_shortcut.history_attempt",
                    {
                        "tool": routed_tool,
                        "chat_history_turns": len(chat_history),
                        "history_context_preview": _trim(_history_context(chat_history), 320),
                    },
                )
                shortcut = _run_tool_from_history(routed_tool, query, chat_history)
                if shortcut is not None:
                    _add_event(
                        tool_span,
                        "qa.tool_shortcut.history",
                        {"tool": routed_tool, "source": "chat_history"},
                    )
                    replan = _orchestrator_replan_after_step(
                        query_text=query,
                        plan=plan,
                        step_name="history_shortcut",
                        provisional_answer=str(shortcut.get("answer", "")),
                        tool_result=shortcut.get("tool_result"),
                        pages_used_count=0,
                        citations_count=0,
                    )
                    _add_event(
                        tool_span,
                        "qa.orchestrator.replan.history_shortcut",
                        {k: str(v) for k, v in replan.items()},
                    )
                    _emit_ui_updates(replan.get("ui_updates"))
                    if bool(replan.get("finalize_now", True)):
                        _save_turn(
                            str(shortcut.get("answer", "")),
                            citations=shortcut.get("citations", []) if isinstance(shortcut.get("citations"), list) else [],
                            images=shortcut.get("images", []) if isinstance(shortcut.get("images"), list) else [],
                            tool_result=shortcut.get("tool_result") if isinstance(shortcut.get("tool_result"), dict) else None,
                            source_file=str(shortcut.get("source_file", "chat_history")),
                        )
                        return shortcut
                    _apply_replan_directive(replan)
                    force_continue_page_level = True
                _add_event(
                    tool_span,
                    "qa.tool_shortcut.history_miss",
                    {
                        "tool": routed_tool,
                        "reason": "insufficient_history_context",
                    },
                )
        allow_image_early_answer = bool(plan.get("allow_image_early_answer", True))
        if allow_image_early_answer and effective_image_bytes and routed_tool == "count":
            _emit("Run image-first count tool")
            with traced_span("qa.image_first.count", query=_trim(query, 220)) as image_count_span:
                from cad_pipeline.tools.count_tool import run_count_tool

                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                try:
                    tmp.write(effective_image_bytes)
                    tmp.close()
                    tool_result = run_count_tool(query=query, image_path=tmp.name)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

                count_value = tool_result.get("count")
                details = str(tool_result.get("details", "") or "")
                _set_attr(image_count_span, "count_value", count_value if count_value is not None else "")
                _set_attr(image_count_span, "mode", str(tool_result.get("mode", "")))
                _set_attr(image_count_span, "details_preview", _trim(details, 220))

                answer = details.strip() or (str(count_value) if count_value is not None else "")
                replan = _orchestrator_replan_after_step(
                    query_text=query,
                    plan=plan,
                    step_name="image_first_count",
                    provisional_answer=answer,
                    tool_result=tool_result,
                    pages_used_count=0,
                    citations_count=0,
                )
                _add_event(image_count_span, "qa.orchestrator.replan.image_first_count", {k: str(v) for k, v in replan.items()})
                _emit_ui_updates(replan.get("ui_updates"))
                if bool(replan.get("finalize_now", True)):
                    _save_turn(answer, tool_result=tool_result, source_file="uploaded_image")
                    return {
                        "answer": answer,
                        "pages_used": [],
                        "citations": [],
                        "images": [],
                        "tool_result": tool_result,
                        "source_file": "uploaded_image",
                        "query_type": "qa",
                    }
                _apply_replan_directive(replan)
                force_continue_page_level = True
        if allow_image_early_answer and effective_image_bytes and routed_tool == "area":
            _emit("Run image-first area tool")
            with traced_span("qa.image_first.area", query=_trim(query, 220)) as image_area_span:
                from cad_pipeline.tools.area_tool import run_area_tool

                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                try:
                    tmp.write(effective_image_bytes)
                    tmp.close()
                    tool_result = run_area_tool(query=query, image_path=tmp.name)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

                area_value = tool_result.get("area")
                if area_value in (None, "", "unknown"):
                    area_value = tool_result.get("total_m2")
                details = str(tool_result.get("details", "") or "")
                _set_attr(image_area_span, "area_value", str(area_value if area_value is not None else ""))
                _set_attr(image_area_span, "mode", str(tool_result.get("mode", "")))
                _set_attr(image_area_span, "details_preview", _trim(details, 220))

                answer = details.strip() or (str(area_value) if area_value not in (None, "", "unknown") else "")
                replan = _orchestrator_replan_after_step(
                    query_text=query,
                    plan=plan,
                    step_name="image_first_area",
                    provisional_answer=answer,
                    tool_result=tool_result,
                    pages_used_count=0,
                    citations_count=0,
                )
                _add_event(image_area_span, "qa.orchestrator.replan.image_first_area", {k: str(v) for k, v in replan.items()})
                _emit_ui_updates(replan.get("ui_updates"))
                if bool(replan.get("finalize_now", True)):
                    _save_turn(answer, tool_result=tool_result, source_file="uploaded_image")
                    return {
                        "answer": answer,
                        "pages_used": [],
                        "citations": [],
                        "images": [],
                        "tool_result": tool_result,
                        "source_file": "uploaded_image",
                        "query_type": "qa",
                    }
                _apply_replan_directive(replan)
                force_continue_page_level = True
        if allow_image_early_answer and effective_image_bytes and routed_tool == "none":
            _emit("Run image-first general vision QA")
            with traced_span("qa.image_first.general", query=_trim(query, 220)) as image_general_span:
                try:
                    from google import genai  # type: ignore
                    from google.genai import types as _gt  # type: ignore
                    from cad_pipeline.config import GEMINI_API_KEY, GEMINI_PRO_MODEL

                    client = genai.Client(api_key=GEMINI_API_KEY)
                    prompt = (
                        "You are an assistant analyzing the user's uploaded CAD/drawing image.\n"
                        "Answer the user's question using ONLY the uploaded image.\n"
                        "If the image does not contain enough evidence, say that clearly.\n"
                        "Do not rely on external document/page context.\n\n"
                        f'User question: "{query}"'
                    )
                    resp = client.models.generate_content(
                        model=GEMINI_PRO_MODEL,
                        contents=[
                            _gt.Part.from_bytes(data=effective_image_bytes, mime_type="image/png"),
                            prompt,
                        ],
                    )
                    answer = str(getattr(resp, "text", "") or "").strip()
                    if not answer:
                        answer = _msg(
                            "Mình chưa đọc được đủ thông tin từ ảnh bạn gửi để trả lời chính xác.",
                            "アップロード画像から十分な情報を読み取れず、正確に回答できませんでした。",
                            "I could not extract enough information from your uploaded image to answer accurately.",
                        )
                    tool_result = {"mode": "vision_pro_qa", "source": "uploaded_image"}
                    _set_attr(image_general_span, "mode", "vision_pro_qa")
                    _set_attr(image_general_span, "answer_preview", _trim(answer, 220))
                    replan = _orchestrator_replan_after_step(
                        query_text=query,
                        plan=plan,
                        step_name="image_first_general",
                        provisional_answer=answer,
                        tool_result=tool_result,
                        pages_used_count=0,
                        citations_count=0,
                    )
                    _add_event(image_general_span, "qa.orchestrator.replan.image_first_general", {k: str(v) for k, v in replan.items()})
                    _emit_ui_updates(replan.get("ui_updates"))
                    if bool(replan.get("finalize_now", True)):
                        _save_turn(answer, tool_result=tool_result, source_file="uploaded_image")
                        return {
                            "answer": answer,
                            "pages_used": [],
                            "citations": [],
                            "images": [],
                            "tool_result": tool_result,
                            "source_file": "uploaded_image",
                            "query_type": "qa",
                        }
                    _apply_replan_directive(replan)
                    force_continue_page_level = True
                except Exception as exc:
                    _set_attr(image_general_span, "error", _trim(str(exc), 180))
        force_page_level = bool(plan.get("force_page_level", False)) or (
            routed_tool in {"count", "area", "viz", "search", "report_pdf", "report_excel"}
        )
        if isinstance(flow_overrides.get("force_page_level"), bool):
            force_page_level = bool(flow_overrides["force_page_level"])
        if force_continue_page_level:
            force_page_level = True
        _set_attr(root_span, "routed_tool", routed_tool)
        _set_attr(root_span, "force_page_level", force_page_level)

        _emit("Load folder and files")
        with traced_span("qa.load_folder_files", folder_id=folder_id) as folder_span:
            folder = mongo.get_folder(folder_id)
            if session_file_ids is not None:
                # Session-scoped chat can include files across multiple folders.
                all_files = mongo.get_files_by_ids(session_file_ids)
            else:
                if not folder:
                    return {"answer": _msg(f"Không tìm thấy thư mục '{folder_id}'.", f"フォルダ '{folder_id}' が見つかりません。", f"Folder '{folder_id}' not found."), "pages_used": [], "images": []}
                all_files = mongo.list_files(folder_id)
            _set_attr(folder_span, "folder_name", str((folder or {}).get("folder_name", "")))
            _set_attr(folder_span, "folder_summary_preview", _trim(str((folder or {}).get("summary", "")), 260))
            _set_attr(folder_span, "file_count", len(all_files))
            _set_attr(folder_span, "file_name_sample", ", ".join((f.get("file_name", "") for f in all_files[:5])))
            _set_attr(root_span, "folder_name", str((folder or {}).get("folder_name", "")))
            _set_attr(root_span, "file_count", len(all_files))
            if not all_files:
                return {"answer": _msg("Không có tài liệu trong thư mục này.", "このフォルダにはファイルがありません。", "No files found in this folder."), "pages_used": [], "images": []}

        folder_name = str((folder or {}).get("name") or folder_id)
        raw_folder_summary = str((folder or {}).get("summary", "") or "")
        # folder.summary can be an aggregated list of file short summaries.
        # Keep folder context concise to avoid duplicating the same information
        # already provided in `files` to folder_agent.
        effective_folder_summary = f"Folder: {folder_name}\nFile count: {len(all_files)}"
        if raw_folder_summary and len(raw_folder_summary) <= 400:
            effective_folder_summary += f"\nNotes: {raw_folder_summary}"
        if session_file_ids is not None:
            _emit("Apply chat-session source scope")
            with traced_span("qa.apply_session_scope", selected_files=len(session_file_ids)) as scope_span:
                allowed = set(session_file_ids)
                all_files = [f for f in all_files if f["_id"] in allowed]
                _set_attr(scope_span, "scoped_file_count", len(all_files))
                _set_attr(scope_span, "scoped_file_sample", ", ".join((f.get("file_name", "") for f in all_files[:5])))
                _set_attr(root_span, "scoped_file_count", len(all_files))
            if not all_files:
                return {
                    "answer": _msg("Không tìm thấy nội dung trang phù hợp cho câu hỏi này.", "この質問に対応するページ内容が見つかりません。", "No page content found for this query."),
                    "pages_used": [],
                    "images": [],
                    "tool_result": None,
                    "source_file": "",
                    "query_type": "qa",
                }
            # Do not rebuild folder summary from file summaries here; the file list
            # already carries short summaries for folder-agent selection.
        _set_attr(root_span, "effective_context_preview", _trim(effective_folder_summary, 300))

        allow_folder_direct_answer = bool(plan.get("allow_folder_direct_answer", True))
        allow_file_direct_answer = bool(plan.get("allow_file_direct_answer", True))
        if isinstance(flow_overrides.get("allow_folder_direct_answer"), bool):
            allow_folder_direct_answer = bool(flow_overrides["allow_folder_direct_answer"])
        if isinstance(flow_overrides.get("allow_file_direct_answer"), bool):
            allow_file_direct_answer = bool(flow_overrides["allow_file_direct_answer"])

        if file_id:
            target_file_ids = [file_id]
        else:
            _emit("Run folder agent")
            with traced_span(
                "qa.folder_agent",
                file_count=len(all_files),
                context_preview=_trim(effective_folder_summary, 260),
            ) as folder_agent_span:
                folder_result = run_folder_agent(
                    query=query,
                    files=[
                        {
                            "file_id": f["_id"],
                            "file_name": f.get("file_name", ""),
                            "summary": f.get("short_summary") or f.get("summary", ""),
                        }
                        for f in all_files
                    ],
                )
                _set_attr(folder_agent_span, "action", str(folder_result.get("action", "")))
                _set_attr(folder_agent_span, "selected_file_count", len(folder_result.get("file_ids", []) or []))
                _set_attr(folder_agent_span, "answer_preview", _trim(str(folder_result.get("answer", "")), 200))
            if allow_folder_direct_answer and (not force_page_level) and folder_result.get("action") == "answer" and folder_result.get("answer"):
                answer = folder_result["answer"]
                replan = _orchestrator_replan_after_step(
                    query_text=query,
                    plan=plan,
                    step_name="folder_direct_answer",
                    provisional_answer=answer,
                    tool_result=None,
                    pages_used_count=0,
                    citations_count=0,
                )
                _add_event(folder_agent_span, "qa.orchestrator.replan.folder_answer", {k: str(v) for k, v in replan.items()})
                _emit_ui_updates(replan.get("ui_updates"))
                if bool(replan.get("finalize_now", True)):
                    _save_turn(answer, source_file="folder_summary")
                    return {
                        "answer": answer,
                        "pages_used": [],
                        "images": [],
                        "tool_result": None,
                        "source_file": "folder_summary",
                        "query_type": "qa",
                    }
                _apply_replan_directive(replan)
                force_page_level = True
            selected_file_ids = folder_result.get("file_ids", []) or []
            if selected_file_ids:
                target_file_ids = selected_file_ids
            elif force_page_level:
                # For tool/page-level queries, avoid broad file-agent fanout only when
                # folder-level cannot narrow files.
                target_file_ids = [f["_id"] for f in all_files]
            else:
                target_file_ids = [f["_id"] for f in all_files[:2]]
        _set_attr(root_span, "target_file_count", len(target_file_ids))
        _set_attr(root_span, "target_file_ids_preview", ",".join(target_file_ids[:10]))

        _emit("Run file agent and collect candidate pages")
        final_page_pool: list[dict] = []
        source_file_name = ""
        for fid in target_file_ids:
            with traced_span("qa.file_loop", file_id=fid) as file_loop_span:
                file_doc = mongo.get_file(fid)
                if not file_doc:
                    continue
                source_file_name = file_doc.get("file_name", fid)
                _set_attr(file_loop_span, "file_name", source_file_name)
                file_short_summary = str(file_doc.get("short_summary") or "")
                file_detailed_summary = str(file_doc.get("summary") or "")
                _set_attr(file_loop_span, "file_short_summary_preview", _trim(file_short_summary, 180))
                _set_attr(file_loop_span, "file_summary_preview", _trim(file_detailed_summary, 220))
                candidate_pages_from_file: list[int] = []
                if not force_page_level:
                    with traced_span("qa.file_agent", file_id=fid, file_name=source_file_name, query=_trim(query, 180)) as file_agent_span:
                        file_result = run_file_agent(
                            query=query,
                            file_id=fid,
                            file_name=source_file_name,
                            file_short_summary=file_short_summary,
                            file_summary=file_detailed_summary,
                        )
                        _set_attr(file_agent_span, "action", str(file_result.get("action", "")))
                        _set_attr(file_agent_span, "page_candidates_count", len(file_result.get("candidate_pages", []) or []))
                        _set_attr(file_agent_span, "answer_preview", _trim(str(file_result.get("answer", "")), 180))
                    if allow_file_direct_answer and file_result.get("action") == "answer" and file_result.get("answer"):
                        answer = file_result["answer"]
                        replan = _orchestrator_replan_after_step(
                            query_text=query,
                            plan=plan,
                            step_name="file_direct_answer",
                            provisional_answer=answer,
                            tool_result=None,
                            pages_used_count=0,
                            citations_count=0,
                        )
                        _add_event(file_agent_span, "qa.orchestrator.replan.file_answer", {k: str(v) for k, v in replan.items()})
                        _emit_ui_updates(replan.get("ui_updates"))
                        if bool(replan.get("finalize_now", True)):
                            _save_turn(answer, source_file=source_file_name)
                            return {
                                "answer": answer,
                                "pages_used": [],
                                "images": [],
                                "tool_result": None,
                                "source_file": source_file_name,
                                "query_type": "qa",
                            }
                        _apply_replan_directive(replan)
                        force_page_level = True
                    candidate_pages_raw = file_result.get("candidate_pages", []) or []
                    candidate_pages_from_file = sorted(
                        {
                            int(p)
                            for p in candidate_pages_raw
                            if isinstance(p, int) and p > 0
                        }
                    )
                else:
                    _add_event(
                        file_loop_span,
                        "qa.file_agent.skipped",
                        {"reason": "force_page_level", "file_id": fid, "file_name": source_file_name},
                    )
                with traced_span("qa.load_pages_by_file", file_id=fid, file_name=source_file_name) as pages_span:
                    pages = mongo.get_pages_by_file(
                        fid,
                        projection={
                            "_id": 1,
                            "file_id": 1,
                            "page_number": 1,
                            "context_md": 1,
                            "image_url": 1,
                            "short_summary": 1,
                            "dxf_path": 1,
                        },
                        page_numbers=candidate_pages_from_file or None,
                    )
                    _set_attr(pages_span, "pages_loaded_count", len(pages))
                    _set_attr(pages_span, "candidate_pages_count", len(candidate_pages_from_file))
                    _set_attr(
                        pages_span,
                        "candidate_pages_preview",
                        ",".join(str(p) for p in candidate_pages_from_file[:12]),
                    )
                    if candidate_pages_from_file and not pages:
                        _add_event(
                            pages_span,
                            "qa.candidate_pages.empty_fallback_all",
                            {
                                "file_id": fid,
                                "candidate_pages": ",".join(str(p) for p in candidate_pages_from_file[:20]),
                            },
                        )
                        pages = mongo.get_pages_by_file(
                            fid,
                            projection={
                                "_id": 1,
                                "file_id": 1,
                                "page_number": 1,
                                "context_md": 1,
                                "image_url": 1,
                                "short_summary": 1,
                                "dxf_path": 1,
                            },
                        )
                        _set_attr(pages_span, "pages_loaded_count_after_fallback", len(pages))
                for p in pages:
                    p["page_id"] = str(p.pop("_id", p.get("page_id", "")))
                    p["file_id"] = p.get("file_id", fid)
                    p["file_name"] = file_doc.get("file_name", fid)
                final_page_pool.extend(pages)

        if not final_page_pool:
            return {
                "answer": _msg("Không tìm thấy nội dung trang phù hợp cho câu hỏi này.", "この質問に対応するページ内容が見つかりません。", "No page content found for this query."),
                "pages_used": [],
                "images": [],
                "tool_result": None,
                "source_file": source_file_name,
                "query_type": "qa",
            }

        _emit("Run page agent and tool execution")
        with traced_span(
            "qa.page_agent",
            page_pool_size=len(final_page_pool),
            query=_trim(query, 240),
            routed_tool=routed_tool,
        ) as page_span:
            page_result = run_page_agent(
                query=query,
                pages=final_page_pool,
                use_tools=True,
                chat_history=chat_history,
                folder_id=folder_id,
                file_id=file_id,
                image_bytes=effective_image_bytes,
                answer_stream_callback=answer_stream_callback,
            )
            _set_attr(page_span, "pages_used_count", len(page_result.get("pages_used", []) or []))
            _set_attr(page_span, "citations_count", len(page_result.get("citations", []) or []))
            _set_attr(page_span, "tool_name", str((page_result.get("tool_result") or {}).get("tool_name", "")))
            _set_attr(page_span, "answer_preview", _trim(str(page_result.get("answer", "")), 220))
            _add_event(
                page_span,
                "qa.page_agent.completed",
                {
                    "pages_used": ",".join(str(p) for p in (page_result.get("pages_used", []) or [])[:30]),
                    "citations_count": len(page_result.get("citations", []) or []),
                },
            )
        answer = page_result.get("answer", "")

        if save_history and answer:
            _emit("Save chat history")
            with traced_span(
                "qa.save_chat_history",
                folder_id=folder_id,
                query=_trim(query, 180),
                answer_preview=_trim(answer, 220),
                answer_length=len(answer),
            ) as save_span:
                _save_turn(
                    answer,
                    citations=page_result.get("citations", []) if isinstance(page_result.get("citations"), list) else [],
                    images=page_result.get("images", []) if isinstance(page_result.get("images"), list) else [],
                    tool_result=page_result.get("tool_result") if isinstance(page_result.get("tool_result"), dict) else None,
                    source_file=source_file_name,
                )
                _add_event(save_span, "qa.chat_history.saved", {"user_email": user_email or "", "answer_length": len(answer)})

        _emit("Finalize response")
        citations = page_result.get("citations", []) or []
        if not citations:
            used_pages = set(page_result.get("pages_used", []) or [])
            if used_pages:
                seen: set[tuple[str, int]] = set()
                for p in final_page_pool:
                    pno = p.get("page_number")
                    fid = str(p.get("file_id", ""))
                    if pno not in used_pages or not fid:
                        continue
                    key = (fid, int(pno))
                    if key in seen:
                        continue
                    seen.add(key)
                    citations.append(
                        {
                            "file_id": fid,
                            "file_name": str(p.get("file_name") or fid),
                            "page_number": int(pno),
                        }
                    )
        _set_attr(root_span, "final_pages_used_count", len(page_result.get("pages_used", []) or []))
        _set_attr(root_span, "final_citations_count", len(citations))
        _set_attr(root_span, "final_source_file", source_file_name)
        _add_event(
            root_span,
            "qa.response.ready",
            {
                "source_file": source_file_name,
                "pages_used": ",".join(str(p) for p in (page_result.get("pages_used", []) or [])[:30]),
                "citations_count": len(citations),
            },
        )

        return {
            "answer": answer,
            "pages_used": page_result.get("pages_used", []),
            "citations": citations,
            "images": page_result.get("images", []),
            "tool_result": page_result.get("tool_result"),
            "source_file": source_file_name,
            "query_type": "qa",
        }
