"""Backfill file.title_block_index from existing pages.blocks.

Usage:
  python -m cad_pipeline.scripts.backfill_title_block_index
  python -m cad_pipeline.scripts.backfill_title_block_index --folder-id default
  python -m cad_pipeline.scripts.backfill_title_block_index --file-id <file_id>
"""

from __future__ import annotations

import argparse
import re

from cad_pipeline.storage import mongo


def _normalize(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def _extract_entries(page_number: int, blocks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for block in blocks:
        if str(block.get("type", "")) != "title_block":
            continue
        content = block.get("content")
        if not isinstance(content, dict):
            continue
        drawing_no = _normalize(
            content.get("drawing_no")
            or content.get("drawing_number")
            or content.get("sheet_no")
            or content.get("sheet_number")
            or content.get("図面番号")
            or content.get("図番")
        )
        drawing_title = _normalize(
            content.get("drawing_title")
            or content.get("title")
            or content.get("sheet_title")
            or content.get("図面名称")
            or content.get("図面名")
        )
        project = _normalize(
            content.get("project")
            or content.get("project_name")
            or content.get("工事名")
            or content.get("物件名")
        )
        if not (drawing_no or drawing_title or project):
            continue
        rows.append(
            {
                "page_number": int(page_number),
                "drawing_no": drawing_no,
                "drawing_title": drawing_title,
                "project": project,
                "source": "title_block",
                "raw": content,
            }
        )
    return rows


def _backfill_one_file(file_doc: dict) -> int:
    file_id = str(file_doc.get("_id", ""))
    if not file_id:
        return 0
    pages = mongo.get_pages_by_file(file_id, projection={"page_number": 1, "blocks": 1})
    index_rows: list[dict] = []
    for page in pages:
        pno = int(page.get("page_number") or 0)
        if pno <= 0:
            continue
        blocks = page.get("blocks") or []
        if not isinstance(blocks, list):
            continue
        index_rows.extend(_extract_entries(pno, blocks))
    mongo.update_file_title_block_index(file_id, index_rows)
    return len(index_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill title_block_index for files collection.")
    parser.add_argument("--folder-id", default="", help="Optional folder scope")
    parser.add_argument("--file-id", default="", help="Optional single file id")
    args = parser.parse_args()

    if args.file_id:
        file_doc = mongo.get_file(args.file_id)
        if not file_doc:
            print(f"[skip] file not found: {args.file_id}")
            return
        count = _backfill_one_file(file_doc)
        print(f"[ok] {args.file_id}: {count} title-block entries")
        return

    files = mongo.list_files(args.folder_id) if args.folder_id else mongo.get_db()["files"].find()
    total_files = 0
    total_rows = 0
    for file_doc in files:
        total_files += 1
        rows = _backfill_one_file(file_doc)
        total_rows += rows
        print(f"[ok] {file_doc.get('_id')}: {rows} entries")
    print(f"Done. files={total_files}, title_block_rows={total_rows}")


if __name__ == "__main__":
    main()

