"""qa_pipeline.py — Full Q&A pipeline: Router → Folder → File → Page → Tool.

Usage:
  from cad_pipeline.pipeline.qa_pipeline import run_qa
  result = run_qa(query="バルブは何個ありますか？", folder_id="folder_001")
"""

from __future__ import annotations

import json
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
    progress_callback: Callable[[str], None] | None = None,
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
    def _emit(step: str) -> None:
        if progress_callback is not None:
            progress_callback(step)

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

    lang = detect_query_language(query)

    def _msg(vi: str, ja: str, en: str) -> str:
        return vi if lang == "vi" else ja if lang == "ja" else en

    def _history_context(turns: list[dict], limit: int = 5) -> str:
        lines: list[str] = []
        for t in turns[-limit:]:
            lines.append(f"User: {t.get('role_user', '')}")
            lines.append(f"Assistant: {t.get('role_assistant', '')}")
        return "\n".join(lines)

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

    def _decide_image_grounding(query_text: str, turns: list[dict]) -> dict[str, str | bool]:
        """Use LLM semantics to decide whether this turn should stay grounded on uploaded image."""
        if not query_text.strip():
            return {"use_uploaded_image": True, "confidence": "low", "reason": "empty_query"}

        history_text = _history_context(turns, limit=6)
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""You are a routing judge in a CAD chat system.

Context:
- The current turn has an uploaded image available from the user.
- The assistant may answer from either:
  A) the uploaded image context, OR
  B) document/page/file context in the knowledge base.

Recent conversation:
---
{history_text}
---

Current user query:
"{query_text}"

Decide whether this query should be grounded on the uploaded image.

Decision rules:
- Choose true if the user is asking about visual/layout details that are best answered from the uploaded image.
- Choose false if the user intent is to return to document/page/file scope, citation scope, or broader document QA.
- If ambiguous, prefer true only when image-grounded interpretation is more plausible from conversation continuity.

Reply ONLY as JSON:
{{
  "use_uploaded_image": true | false,
  "confidence": "high" | "medium" | "low",
  "reason": "<short reason>"
}}"""
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            raw = str(getattr(resp, "text", "") or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            use_uploaded = bool(parsed.get("use_uploaded_image", True))
            confidence = str(parsed.get("confidence", "low")).lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            reason = _trim(str(parsed.get("reason", "")), 180) or "model_decision"
            return {
                "use_uploaded_image": use_uploaded,
                "confidence": confidence,
                "reason": reason,
            }
        except Exception as exc:
            return {
                "use_uploaded_image": True,
                "confidence": "low",
                "reason": f"fallback_on_error:{_trim(str(exc), 80)}",
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

        effective_image_bytes = image_bytes
        if image_bytes:
            with traced_span("qa.image_scope_decision", query=_trim(query, 220)) as image_scope_span:
                image_scope = _decide_image_grounding(query, chat_history)
                use_uploaded_image = bool(image_scope.get("use_uploaded_image", True))
                if not use_uploaded_image:
                    effective_image_bytes = None
                _set_attr(image_scope_span, "use_uploaded_image", use_uploaded_image)
                _set_attr(image_scope_span, "confidence", str(image_scope.get("confidence", "")))
                _set_attr(image_scope_span, "reason", str(image_scope.get("reason", "")))
                _set_attr(root_span, "use_uploaded_image", use_uploaded_image)
                _add_event(
                    image_scope_span,
                    "qa.image_scope.selected",
                    {
                        "use_uploaded_image": use_uploaded_image,
                        "confidence": str(image_scope.get("confidence", "")),
                        "reason": str(image_scope.get("reason", "")),
                    },
                )

        _emit("Detect query tool intent")
        with traced_span("qa.detect_tool_intent", query=_trim(query, 240)) as tool_span:
            routed_tool = classify_tool(query=query, image_bytes=effective_image_bytes).get("tool", "none")
            _set_attr(tool_span, "routed_tool", routed_tool)
            # Do not use history shortcut when user provided an image.
            # Image queries must be grounded on the uploaded image input.
            if (not effective_image_bytes) and routed_tool in {"count", "area", "report_pdf", "report_excel"}:
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
                    if save_history and shortcut.get("answer"):
                        mongo.append_chat_turn(folder_id, query, str(shortcut["answer"]), user_email=user_email)
                    return shortcut
                _add_event(
                    tool_span,
                    "qa.tool_shortcut.history_miss",
                    {
                        "tool": routed_tool,
                        "reason": "insufficient_history_context",
                    },
                )
        # Image-first hard guard:
        # For count/area queries with uploaded image, answer directly from the uploaded image.
        # Do not run page-level reasoning to avoid mixing unrelated page context.
        if effective_image_bytes and routed_tool == "count":
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

                if save_history and answer:
                    mongo.append_chat_turn(folder_id, query, answer, user_email=user_email)

                return {
                    "answer": answer,
                    "pages_used": [],
                    "citations": [],
                    "images": [],
                    "tool_result": tool_result,
                    "source_file": "uploaded_image",
                    "query_type": "qa",
                }
        if effective_image_bytes and routed_tool == "area":
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

                if save_history and answer:
                    mongo.append_chat_turn(folder_id, query, answer, user_email=user_email)

                return {
                    "answer": answer,
                    "pages_used": [],
                    "citations": [],
                    "images": [],
                    "tool_result": tool_result,
                    "source_file": "uploaded_image",
                    "query_type": "qa",
                }
        if effective_image_bytes and routed_tool == "none":
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

                    if save_history and answer:
                        mongo.append_chat_turn(folder_id, query, answer, user_email=user_email)

                    return {
                        "answer": answer,
                        "pages_used": [],
                        "citations": [],
                        "images": [],
                        "tool_result": tool_result,
                        "source_file": "uploaded_image",
                        "query_type": "qa",
                    }
                except Exception as exc:
                    _set_attr(image_general_span, "error", _trim(str(exc), 180))
        force_page_level = routed_tool in {"count", "area", "viz", "search", "report_pdf", "report_excel"}
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
            if (not force_page_level) and folder_result.get("action") == "answer" and folder_result.get("answer"):
                answer = folder_result["answer"]
                if save_history:
                    mongo.append_chat_turn(folder_id, query, answer, user_email=user_email)
                return {
                    "answer": answer,
                    "pages_used": [],
                    "images": [],
                    "tool_result": None,
                    "source_file": "folder_summary",
                    "query_type": "qa",
                }
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
                    if file_result.get("action") == "answer" and file_result.get("answer"):
                        answer = file_result["answer"]
                        if save_history:
                            mongo.append_chat_turn(folder_id, query, answer, user_email=user_email)
                        return {
                            "answer": answer,
                            "pages_used": [],
                            "images": [],
                            "tool_result": None,
                            "source_file": source_file_name,
                            "query_type": "qa",
                        }
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
                mongo.append_chat_turn(folder_id, query, answer, user_email=user_email)
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
