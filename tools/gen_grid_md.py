#!/usr/bin/env python3
"""gen_grid_md.py — Generate structured Markdown from DXF spec sheets.

Supports:
  D003 — 特記仕様書（1）　共通  (narrative + 工事区分 checkbox tables)
  D004 — 特記仕様書（2）　共通  (system work-division diagrams)

Usage:
  python3 gen_grid_md.py <file.dxf>            # auto-detects type
  python3 gen_grid_md.py <file.dxf> -o out.md  # explicit output path
  python3 gen_grid_md.py --help
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ezdxf

from grid_engine import (
    _extract_texts,
    _extract_lines,
    build_grid,
    assign_texts,
    snap_circles_to_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
# Title block extraction (表題欄)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_title_block(doc: ezdxf.document.Drawing, msp) -> dict:
    """Extract drawing metadata from 図面枠 + 01_建築者氏名 blocks and modelspace strip."""
    info: dict[str, str] = {}

    # ── 図面枠 block ──────────────────────────────────────────────────────
    waku = doc.blocks.get("図面枠")
    if waku:
        for e in waku:
            if e.dxftype() != "TEXT":
                continue
            t = e.dxf.text.strip()
            y = e.dxf.insert.y
            x = e.dxf.insert.x
            if "新築工事" in t or "工事" in t and x > 600:
                info.setdefault("工事名", t)
            elif re.search(r"\d{4}\.\d{2}\.\d{2}", t):
                info.setdefault("日付", t)
            elif "完" in t and "図" in t:
                info.setdefault("図面状態", re.sub(r"\s+", "", t))
            elif re.search(r"事.{0,2}務.{0,2}所", t):
                cleaned = re.sub(r"\s+", "", t)   # remove all spaces in spaced kanji
                info.setdefault("設計事務所", cleaned)
            elif "TEL" in t or "丁目" in t:
                info.setdefault("所在地", t)

    # ── 01_建築者氏名 block ───────────────────────────────────────────────
    architects: list[str] = []
    kens = doc.blocks.get("01_建築者氏名")
    if kens:
        for e in kens:
            if e.dxftype() == "TEXT" and e.dxf.text.strip():
                architects.append(e.dxf.text.strip())
    if architects:
        info["一級建築士"] = " / ".join(architects)

    # ── Modelspace bottom strip (Y < 40) ─────────────────────────────────
    all_texts = _extract_texts(msp)
    strip = sorted([t for t in all_texts if t.y < 40], key=lambda t: t.x)
    sheet_num = ""
    sheet_title = ""
    for t in strip:
        if t.layer == "MOJI1":
            raw = t.text.strip()
            if len(raw) <= 3 and re.match(r"^[\dＤ\-]+$", raw):
                sheet_num = raw
            elif len(raw) > 3:
                sheet_title = raw
        elif t.layer == "WAKU-MOJI" and t.text.strip() not in ("", "-"):
            sheet_title = t.text.strip()
    # Drawing number: 'D' prefix + sheet_num from strip
    if sheet_num:
        info["図面番号"] = f"D{sheet_num.zfill(3)}"
    if sheet_title:
        info["図面名"] = sheet_title

    return info


def _title_block_md(info: dict) -> str:
    if not info:
        return ""
    order = ["図面番号", "図面名", "工事名", "図面状態", "日付", "設計事務所", "所在地", "一級建築士"]
    rows = [(k, info[k]) for k in order if k in info]
    rows += [(k, v) for k, v in info.items() if k not in order]
    lines = ["## 図面情報\n", "| 項目 | 内容 |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def group_by_y(items: list, tol: float = 5.0) -> list[list]:
    """Cluster items into rows by Y-proximity (descending Y order)."""
    rows: list[list] = []
    for t in sorted(items, key=lambda t: -t.y):
        placed = False
        for r in rows:
            if abs(r[0].y - t.y) <= tol:
                r.append(t)
                placed = True
                break
        if not placed:
            rows.append([t])
    return sorted(rows, key=lambda r: -r[0].y)


def _is_number_text(s: str) -> bool:
    FW = set("０１２３４５６７８９")
    s = s.strip()
    return bool(s) and all(c in FW or c.isdigit() for c in s) and len(s) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# D003 generator
# ─────────────────────────────────────────────────────────────────────────────

_HDR_RE = re.compile(r"^\d+[.．][^\d]")

SECTION_TITLES = {
    "建築関係", "冷暖房換気設備関係", "電気設備関係",
    "昇降機設備関係", "給排水設備関係", "機械駐車設備関係",
}
COL_HEADERS = {"建築", "電気", "給排水", "冷暖換", "昇降機", "別途", "機械駐車"}
TRADES_D003 = {"建築", "電気", "給排水", "冷暖換気"}
SUBTABLE_COLS = range(5, 15)


def _gen_d003(dxf_path: Path) -> str:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    h_segs, v_segs = _extract_lines(msp)
    all_texts = _extract_texts(msp)
    grid = build_grid(h_segs, v_segs, cluster_tol=1.5)
    matrix, unassigned = assign_texts(grid, all_texts)

    n_rows = grid.rows
    n_cols = grid.cols

    # Circles
    circles: list[tuple[float, float, float]] = [
        (e.dxf.center.x, e.dxf.center.y, e.dxf.radius)
        for e in msp
        if e.dxftype() == "CIRCLE" and e.dxf.radius < 5
    ]

    def d2m(di: int) -> int:
        return n_rows - 1 - di

    def ct(ri: int, ci: int) -> str:
        if 0 <= ri < n_rows and 0 <= ci < n_cols:
            return " / ".join(matrix[ri][ci])
        return ""

    # Cell text with Y-coords
    cell_text_ys: dict[tuple[int, int], list[tuple[float, str]]] = {}
    for item in all_texts:
        cell = grid.cell_at(item.x, item.y)
        if cell:
            ri, ci = cell
            cell_text_ys.setdefault((ri, ci), []).append((item.y, item.text))

    # Circle cells
    circle_mat: dict[tuple[int, int], bool] = {}
    circle_cell_ys: dict[tuple[int, int], list[float]] = {}
    for cx, cy, _ in circles:
        cell = grid.cell_at(cx, cy)
        if cell:
            ri, ci = cell
            circle_mat[(ri, ci)] = True
            circle_cell_ys.setdefault((ri, ci), []).append(cy)

    # Column layout detection
    dividers_x = sorted(set(
        round((e.dxf.start.x + e.dxf.end.x) / 2, 0)
        for e in msp
        if e.dxftype() == "LINE"
        and abs(e.dxf.start.x - e.dxf.end.x) < 2
        and abs(e.dxf.start.y - e.dxf.end.y) > 200
        and 0 < e.dxf.start.x < 400
    ))

    def col_for_x(x: float) -> int:
        return min(range(len(grid.col_xs) - 1), key=lambda i: abs(grid.col_xs[i] - x))

    div_cols = [col_for_x(x) for x in dividers_x if x < 400]
    if len(div_cols) < 2:
        div_cols = [0, 1, 2]
    narr_left_col = div_cols[0]
    narr_end_col = div_cols[1]

    # Detect heading column
    ci_hdr: int | None = None
    for ci in range(narr_left_col + 1, narr_end_col + 1):
        for ri in range(n_rows):
            if ct(ri, ci) == "工事仕様":
                ci_hdr = ci
                break
        if ci_hdr is not None:
            break
    if ci_hdr is None:
        ci_hdr = narr_left_col + 1

    LEFT_X_MAX = grid.col_xs[narr_left_col + 1]
    KK_COLS = range(narr_end_col + 1, n_cols)

    # ── Left stream (工事概要) ─────────────────────────────────────────────
    SKIP_TEXTS = {"特記仕様書-1（共通）", "工事概要", "一般特記仕様書"}
    SECTION_HDR_RE = re.compile(r"^\d+[..]")

    left_items: list[tuple[float, float, str]] = []
    for u in unassigned:
        if u.x < LEFT_X_MAX:
            left_items.append((u.y, u.x, u.text))
    for item in all_texts:
        cell = grid.cell_at(item.x, item.y)
        if cell and cell[1] == narr_left_col:
            left_items.append((item.y, item.x, item.text))
    left_items.sort(key=lambda t: (-t[0], t[1]))

    def merge_by_row(items: list, y_tol: float = 3.0) -> list[tuple[float, list]]:
        rows: list[tuple[float, list]] = []
        cur: list = []
        cur_y: float | None = None
        for y, x, t in items:
            if cur_y is None or abs(y - cur_y) <= y_tol:
                cur.append((x, t))
                cur_y = cur_y or y
            else:
                rows.append((cur_y, sorted(cur, key=lambda v: v[0])))
                cur = [(x, t)]
                cur_y = y
        if cur:
            rows.append((cur_y, sorted(cur, key=lambda v: v[0])))  # type: ignore[arg-type]
        return rows

    def format_left_row(row_y: float, items: list) -> str | None:
        texts = [t for _, t in items]
        if not texts or texts[0] in SKIP_TEXTS:
            return None
        first_x = items[0][0]
        indent = "  " if first_x > 27 else ""
        if len(texts) == 1:
            return f"{indent}{texts[0]}"
        label = texts[0]
        rest = " ".join(t for t in texts[1:] if t)
        if SECTION_HDR_RE.match(label) or len(label) <= 12:
            return f"{indent}**{label}**: {rest}"
        return f"{indent}{' '.join(texts)}"

    # ── Right stream (一般特記仕様書) ─────────────────────────────────────
    def render_narr_subtable(hdr_ri: int, t_start: int, t_end: int) -> str:
        hdr_cols = [ci for ci in SUBTABLE_COLS if ct(hdr_ri, ci) and "**" not in ct(hdr_ri, ci)]
        if not hdr_cols:
            return ""
        hdrs = [ct(hdr_ri, ci) for ci in hdr_cols]
        rows = []
        for ri in range(t_start, t_end + 1):
            label = ct(ri, narr_end_col)
            if label in TRADES_D003 or any("**" in ct(ri, c) for c in SUBTABLE_COLS):
                marks = ["○" if "**" in ct(ri, ci) else "·" for ci in hdr_cols]
                if label or any(m == "○" for m in marks):
                    rows.append((label or "-", marks))
        if not rows:
            return ""
        lines = ["| | " + " | ".join(hdrs) + " |", "|---|" + "---|" * len(hdrs)]
        for label, marks in rows:
            lines.append(f"| {label} | " + " | ".join(marks) + " |")
        return "\n".join(lines) + "\n"

    def right_section_md() -> str:
        lines: list[str] = []
        ri = 0
        while ri < n_rows:
            hdr = ct(ri, ci_hdr).strip()
            cp = [ct(ri, ci) for ci in range(ci_hdr + 1, narr_end_col + 1) if ct(ri, ci)]
            content = " ".join(cp).strip()
            if hdr and re.fullmatch(r"[**\s]+", hdr):
                hdr = ""
            if content and re.fullmatch(r"[**\s]+", content):
                content = ""
            size_hdrs = [ci for ci in SUBTABLE_COLS if ct(ri, ci) and "**" not in ct(ri, ci)]
            if size_hdrs and not hdr and not content:
                hdr_ri = ri
                ri += 1
                tblocks: list[tuple[int, int]] = []
                cbs = None
                while ri < n_rows:
                    trade = ct(ri, narr_end_col)
                    hm = any("**" in ct(ri, c) for c in SUBTABLE_COLS)
                    if ct(ri, ci_hdr) and not (trade in TRADES_D003 and not ct(ri, ci_hdr).strip()):
                        break
                    if trade in TRADES_D003 or hm:
                        if cbs is None:
                            cbs = ri
                        ri += 1
                    elif cbs is not None:
                        tblocks.append((cbs, ri - 1))
                        cbs = None
                        ri += 1
                    else:
                        ri += 1
                if cbs is not None:
                    tblocks.append((cbs, ri - 1))
                for bs, be in tblocks:
                    tbl = render_narr_subtable(hdr_ri, bs, be)
                    if tbl:
                        lines.append(tbl)
                continue
            if not hdr and not content:
                ri += 1
                continue
            ih = bool(hdr and _HDR_RE.match(hdr))
            if ih:
                lines.append(f"\n### {hdr}\n")
                if content:
                    lines.append(content)
            elif hdr:
                lines.append(f"{hdr}" + (f" {content}" if content else ""))
            elif content:
                lines.append(content)
            ri += 1
        return "\n".join(lines) + "\n"

    # ── 工事区分 tables ────────────────────────────────────────────────────
    kk: list[dict] = []
    for ri in range(n_rows):
        for ci in KK_COLS:
            t = ct(ri, ci)
            if t in SECTION_TITLES:
                ch = [(cj, ct(ri, cj)) for cj in range(ci + 1, min(ci + 8, n_cols))
                      if ct(ri, cj) in COL_HEADERS]
                if ch:
                    # Store as display index (counted from top) so d2m(hdr_row±1)
                    # correctly converts back to matrix row.
                    kk.append({"title": t, "hdr_row": n_rows - 1 - ri,
                                "label_col": ci, "col_hdrs": ch})
    kk.sort(key=lambda s: (s["hdr_row"], s["label_col"]))
    for i, s in enumerate(kk):
        s["end_row"] = next(
            (kk[j]["hdr_row"] for j in range(i + 1, len(kk)) if kk[j]["hdr_row"] > s["hdr_row"]),
            n_rows,
        )

    def circles_for_mark_cell(
        mat_ri: int, mark_ci: int, snap: dict
    ) -> list[float]:
        ys = list(circle_cell_ys.get((mat_ri, mark_ci), []))
        for (src_ri, src_ci), (tgt_ri, tgt_ci) in snap.items():
            if tgt_ri == mat_ri and tgt_ci == mark_ci and src_ri != mat_ri:
                ys.extend(circle_cell_ys.get((src_ri, src_ci), []))
        return ys

    # ── Assemble Markdown ─────────────────────────────────────────────────
    title_info = _extract_title_block(doc, msp)
    md = f"# {dxf_path.stem}\n\n"
    md += _title_block_md(title_info)
    md += "## 工事概要\n\n"
    for row_y, items in merge_by_row(left_items):
        line = format_left_row(row_y, items)
        if line:
            md += f"{line}\n"
    md += "\n"

    md += "## 一般特記仕様書\n\n"
    md += right_section_md()
    md += "\n"

    total_circles = 0
    md += "## 工事区分表\n\n"
    for s in kk:
        lc = s["label_col"]
        mat_top = d2m(s["hdr_row"] + 1)
        mat_bottom = d2m(s["end_row"] - 1)
        rng = (mat_bottom, mat_top + 1)
        snap = snap_circles_to_labels(circle_mat, matrix, [lc], (rng[0], rng[1]))
        hdrs = [h for _, h in s["col_hdrs"]]
        hdr_cis = [ci for ci, _ in s["col_hdrs"]]
        data_rows: list[tuple[str, list[str]]] = []

        for ri in range(mat_top, mat_bottom - 1, -1):
            items_in_cell = cell_text_ys.get((ri, lc), [])
            if not items_in_cell:
                continue
            if len(items_in_cell) == 1:
                _, label = items_in_cell[0]
                if not label or label == s["title"]:
                    continue
                marks = [
                    "○" if any(tgt == (ri, ci) for tgt in snap.values()) else "·"
                    for ci in hdr_cis
                ]
                data_rows.append((label, marks))
            else:
                all_ys = [y for y, _ in items_in_cell]
                for text_y, label in items_in_cell:
                    if not label or label == s["title"]:
                        continue
                    marks = []
                    for ci in hdr_cis:
                        all_c_ys = circles_for_mark_cell(ri, ci, snap)
                        has = any(
                            min(all_ys, key=lambda ty: abs(ty - cy)) == text_y
                            for cy in all_c_ys
                        )
                        marks.append("○" if has else "·")
                    data_rows.append((label, marks))

        sec_circles = sum(m == "○" for _, marks in data_rows for m in marks)
        total_circles += sec_circles
        md += f"\n### {s['title']} ({sec_circles})\n\n"
        md += f"| {s['title']} | " + " | ".join(hdrs) + " |\n"
        md += "|---|" + "---|" * len(hdrs) + "\n"
        for label, marks in data_rows:
            md += f"| {label} | " + " | ".join(marks) + " |\n"

    md += f"\n---\n_合計 ○: {total_circles}_\n"
    return md


# ─────────────────────────────────────────────────────────────────────────────
# D004 generator
# ─────────────────────────────────────────────────────────────────────────────

_D004_TRADE_CODES = {"Ｅ", "Ａ", "Ｐ", "Ｄ", "受", "ＥＬＶ", "ＥＳＬ", "Ｓ", "Ｆ", "A"}
_D004_TRADE_NAMES = {
    "Ｅ": "電気工事", "Ａ": "空調工事", "Ｐ": "衛生工事", "Ｄ": "建築工事",
    "受": "受託工事", "ＥＬＶ": "エレベーター工事", "ＥＳＬ": "エスカレーター工事",
    "Ｓ": "防排煙制御盤", "Ｆ": "防火防煙シャッター制御盤",
}
_LEGEND_ZONE = (1034, 1130, 500, 562)
_ELK_TABLE_ZONE = (1519, 1652, 82, 162)
_BAND_LABELS = [
    "火災・排煙設備",
    "シャッター・消防設備",
    "消火・給排水設備",
    "空調・昇降機・電気錠設備",
    "補給水・制御・給水設備",
]
_EXCLUDE_ELK_SECTIONS = {"雑用水給水設備（加圧給水方式）", "自動制御設備"}


def _gen_d004(dxf_path: Path) -> str:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    texts = _extract_texts(msp)

    def is_legend(t) -> bool:
        x0, x1, y0, y1 = _LEGEND_ZONE
        return x0 < t.x < x1 and y0 < t.y < y1

    def is_elk_table(t) -> bool:
        x0, x1, y0, y1 = _ELK_TABLE_ZONE
        return x0 < t.x < x1 and y0 < t.y < y1

    def texts_in(x_lo, x_hi, y_lo, y_hi,
                 exc_legend=False, exc_elk=False) -> list:
        result = []
        for t in texts:
            if not (x_lo < t.x < x_hi and y_lo < t.y < y_hi):
                continue
            if exc_legend and is_legend(t):
                continue
            if exc_elk and is_elk_table(t):
                continue
            result.append(t)
        return sorted(result, key=lambda t: (-t.y, t.x))

    # Section headers (h≈4.5)
    section_hdrs = sorted(
        [t for t in texts if abs(t.height - 4.5) < 0.2 and t.text != "工事区分"],
        key=lambda t: (-t.y, t.x),
    )

    def y_bands(hdrs: list, tol: float = 10.0) -> list[list]:
        bands: list[list] = []
        for h in sorted(hdrs, key=lambda t: -t.y):
            placed = False
            for b in bands:
                if abs(b[0].y - h.y) < tol:
                    b.append(h)
                    placed = True
                    break
            if not placed:
                bands.append([h])
        return [sorted(b, key=lambda t: t.x) for b in bands]

    bands = sorted(y_bands(section_hdrs), key=lambda b: -b[0].y)
    band_ys = [b[0].y for b in bands]

    # Build sections with midpoint X-bounds
    sections: list[dict] = []
    for bi, band in enumerate(bands):
        y_top = band[0].y + 8
        y_bot = band_ys[bi + 1] - 3 if bi + 1 < len(bands) else -410.0
        for si, sec in enumerate(band):
            x_lo = 1020.0 if si == 0 else (band[si - 1].x + sec.x) / 2
            x_hi = 1840.0 if si == len(band) - 1 else (sec.x + band[si + 1].x) / 2
            sections.append({
                "title": sec.text, "x_lo": x_lo, "x_hi": x_hi,
                "y_top": y_top, "y_bot": y_bot, "band_y": band[0].y,
            })

    # 電気錠 No 1-16 table
    elk_items = texts_in(*_ELK_TABLE_ZONE)
    elk_table_rows: list[tuple[str, str]] = []
    for row in group_by_y(elk_items, tol=4.0):
        no_ts = [t for t in row if _is_number_text(t.text)]
        itm_ts = [t for t in row if t not in no_ts
                  and t.text not in ("項　　目", "電気工事", "建築工事", "Ｎｏ")]
        if no_ts and itm_ts:
            no = no_ts[0].text.strip()
            item = " ".join(t.text for t in sorted(itm_ts, key=lambda t: t.x))
            elk_table_rows.append((no, item))

    # ── Assemble Markdown ─────────────────────────────────────────────────
    title_info = _extract_title_block(doc, msp)
    md = f"# {dxf_path.stem}\n\n"
    md += _title_block_md(title_info)

    # Legend
    md += "## 工事区分記号\n\n| 記号 | 工事区分 |\n|---|---|\n"
    for code, name in _D004_TRADE_NAMES.items():
        md += f"| **{code}** | {name} |\n"
    md += "\n"

    # General notes (extracted from DXF)
    md += "## 一般注記\n\n"
    note_raw = sorted(
        [t for t in texts if abs(t.height - 3.4) < 0.2 and t.layer == "KUTAI-SONOTA"
         and 500 < t.y < 560 and t.x > 1034],
        key=lambda t: (-t.y, t.x),
    )
    note_grps = group_by_y(note_raw, tol=12.0)
    seen_nums: set[str] = set()
    # Numbered fallbacks
    fallbacks = {
        "１": "１　特記なき工事区分は下記による。",
        "２": "２　特記なきの機器の取付及び調整は各々の工事に含む。",
        "３": "３　特記なき外部用接点、端子はこれらが入っている盤、機器側の工事に含むものとする。",
        "４": "４　特記なき配線に於いて、配管及びボックスが必要な場所にはこれら一式も含むものとする。又、盤、機器への継ぎ込み迄は配線工事側に含む。",
    }
    num_to_text: dict[str, str] = {}
    for grp in note_grps:
        combined = " ".join(t.text for t in sorted(grp, key=lambda t: t.x))
        m = re.match(r"^([１２３４])", combined)
        if m:
            num_to_text[m.group(1)] = combined
    for num in ["１", "２", "３", "４"]:
        raw = num_to_text.get(num, "")
        # Use raw only if it doesn't contain trade-name text (legend bleed-in)
        if raw and "工　事" not in raw:
            md += f"{raw}\n\n"
        else:
            md += f"{fallbacks[num]}\n\n"

    # Work division sections
    md += "## 工事区分表\n\n"
    prev_band_y: float | None = None
    for sec in sections:
        if sec["band_y"] != prev_band_y:
            bi = next(i for i, b in enumerate(bands) if b[0].y == sec["band_y"])
            label = _BAND_LABELS[bi] if bi < len(_BAND_LABELS) else ""
            if label:
                md += f"### {label}\n\n"
            prev_band_y = sec["band_y"]

        md += f"#### {sec['title']}\n\n"

        if sec["title"] == "電気錠・スイッチストライク":
            md += "| No | 項目 | 建築工事 | 電気工事 |\n|---|---|---|---|\n"
            for no, item in elk_table_rows:
                md += f"| {no} | {item} | | |\n"
            md += "\n> 注記) 建築工事 = 建具業者(電気錠メーカー)施工範囲\n\n"
            continue

        exc_elk = sec["title"] in _EXCLUDE_ELK_SECTIONS
        items = texts_in(sec["x_lo"], sec["x_hi"], sec["y_bot"], sec["y_top"] - 5,
                         exc_legend=True, exc_elk=exc_elk)

        components = [
            t for t in items
            if t.text not in _D004_TRADE_CODES
            and len(t.text) > 1
            and abs(t.height - 4.5) > 0.1
            and t.text not in ("項　　目", "電気工事", "建築工事", "Ｎｏ", "Ｎｏ.")
            and not _is_number_text(t.text)
            and t.text not in ("受座・箱受", "シリンダー錠", "コネクター（両端）",
                               "コンクリードボックス", "コンクリートボックス",
                               "注記) 建築工事 = 建具業者(電気錠メーカー)施工範囲",
                               "特記仕様書-2　共通")
        ]
        codes_in = [t for t in items if t.text in _D004_TRADE_CODES]

        seen: set[tuple[str, str]] = set()
        for comp in components:
            nearby = sorted(set(
                c.text for c in codes_in
                if abs(c.y - comp.y) < 12 and abs(c.x - comp.x) < 80
            ))
            code_str = "/".join(nearby)
            key = (comp.text, code_str)
            if key in seen:
                continue
            seen.add(key)
            line = f"- {comp.text}"
            if code_str:
                line += f" → **{code_str}**"
            md += line + "\n"
        md += "\n"

    return md


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection & dispatch
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# D005 generator  (特記仕様書 narrative / multi-column spec table)
# ─────────────────────────────────────────────────────────────────────────────

def _gen_d005(dxf_path: Path) -> str:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    all_texts = _extract_texts(msp)

    # ── Column groups: determined by section header (h=3.5) X positions ──
    hdrs = sorted([t for t in all_texts if abs(t.height - 3.5) < 0.1], key=lambda t: t.x)

    # Cluster section headers into column X-bands (gap > 150 = new column)
    col_xs: list[float] = []
    for h in sorted(set(h.x for h in hdrs)):
        if not col_xs or h - col_xs[-1] > 150:
            col_xs.append(h)

    # Use long vertical LINE dividers for column boundaries (more accurate than midpoints)
    h_segs_d5, v_segs_d5 = _extract_lines(msp)
    v_dividers = sorted(set(
        round((s[0] + s[2]) / 2)
        for s in v_segs_d5
        if abs(s[0] - s[2]) < 3 and abs(s[1] - s[3]) > 200
    ))
    all_x = sorted(t.x for t in all_texts)
    x_min, x_max = min(all_x), max(all_x)

    # Build col_bounds: each column spans from previous divider to next, matching col_xs
    col_bounds: list[tuple[float, float]] = []
    for i, cx in enumerate(col_xs):
        # Find nearest divider to the LEFT of cx (before the header)
        left_divs = [d for d in v_dividers if d < cx - 10]
        lo = max(left_divs) + 1 if left_divs else x_min - 5
        # Find nearest divider to the RIGHT of cx+180 (after the 標準番号 sub-col)
        right_divs = [d for d in v_dividers if d > cx + 150]
        hi = min(right_divs) - 1 if right_divs else x_max + 5
        col_bounds.append((lo, hi))

    def texts_in_col(lo: float, hi: float, y_lo: float, y_hi: float) -> list:
        return sorted(
            [t for t in all_texts if lo <= t.x <= hi and y_lo <= t.y <= y_hi],
            key=lambda t: (-t.y, t.x),
        )

    # ── Build sections dict: col_idx → list of {title, y_top, y_bot, items} ──
    title_y: float = max(t.y for t in all_texts if abs(t.height - 50) < 5) if any(
        abs(t.height - 50) < 5 for t in all_texts
    ) else 9999
    # Main table Y range (below title row + gap)
    table_y_hi = 580.0   # just below the horizontal separator
    table_y_lo = 40.0    # above the bottom title strip

    def group_sections(col_idx: int) -> list[dict]:
        lo, hi = col_bounds[col_idx]
        col_hdrs = sorted(
            [t for t in hdrs if lo <= t.x <= hi],
            key=lambda t: -t.y,
        )
        sections: list[dict] = []
        for si, sh in enumerate(col_hdrs):
            y_top = sh.y + 2
            y_bot = col_hdrs[si + 1].y - 1 if si + 1 < len(col_hdrs) else table_y_lo
            raw_items = texts_in_col(lo, hi, y_bot, y_top)
            # Exclude other section header texts (h≈3.5) that bleed into boundary
            items = [t for t in raw_items
                     if not (abs(t.height - 3.5) < 0.1 and t.text != sh.text)]
            sections.append({
                "title": sh.text,
                "hdr_x": sh.x,
                "y_top": y_top,
                "y_bot": y_bot,
                "items": items,
            })
        return sections

    # ── Row reconstruction: merge texts at same Y into logical rows ──
    def rows_from_items(items: list, y_tol: float = 2.5) -> list[list]:
        rows: list[list] = []
        for t in items:
            placed = False
            for r in rows:
                if abs(r[0].y - t.y) <= y_tol:
                    r.append(t)
                    placed = True
                    break
            if not placed:
                rows.append([t])
        return rows

    def render_section(sec: dict, col_lo: float) -> str:  # noqa: ARG001
        clean_title = re.sub(r"^\^L", "", sec["title"])
        out: list[str] = [f"\n### {clean_title}\n"]
        rows = rows_from_items(sec["items"])
        num_re = re.compile(r"^\d+[.．]")
        hdr_x: float = sec["hdr_x"]   # section title X — numbered items are near here
        title_clean = re.sub(r"^\^L", "", sec["title"])

        # Buffer for current numbered item
        cur_label: str = ""
        cur_body: list[str] = []
        cur_ref: str = ""

        def flush_item() -> None:
            nonlocal cur_label, cur_body, cur_ref
            if cur_label or cur_body:
                body = " ".join(cur_body).strip()
                ref_str = f"  `{cur_ref}`" if cur_ref else ""
                if cur_label:
                    out.append(f"**{cur_label}** {body}{ref_str}".rstrip())
                else:
                    out.append(f"{body}{ref_str}".rstrip())
            cur_label = ""
            cur_body = []
            cur_ref = ""

        for row in rows:
            ts = sorted(row, key=lambda t: t.x)
            ref_ts = [t for t in ts if t.layer == "MOJI_標準仕様書番号"]
            cont_ts = [t for t in ts if t.layer != "MOJI_標準仕様書番号"]

            ref = " / ".join(t.text.strip() for t in ref_ts) if ref_ts else ""

            if not cont_ts:
                if ref:
                    cur_ref = (cur_ref + " " + ref).strip() if cur_ref else ref
                continue

            left = cont_ts[0]
            lx = left.x
            all_txt = " ".join(t.text for t in cont_ts).strip()

            # Skip header-row items (column labels) and section-title repeat
            skip_texts = {title_clean, "建築工事", "施工場所", "種別", "仕様", "保証年限",
                          "使用箇所", "帳壁の種類", "壁厚", "工法・厚み・層間変位強度・耐風圧等加筆の必要"}
            if all_txt in skip_texts:
                continue

            # Section title repeated as item → skip
            if all_txt == title_clean or all_txt.startswith(title_clean + " "):
                if ref:
                    cur_ref = ref
                continue

            is_numbered = bool(num_re.match(left.text)) and abs(lx - hdr_x) < 10
            is_label_only = (abs(lx - hdr_x) < 10 and not all_txt.startswith("・")
                             and not num_re.match(all_txt) and len(all_txt) < 30
                             and all_txt and not all_txt[0].isdigit())

            if is_numbered:
                flush_item()
                cur_label = left.text
                cur_body = [t.text for t in cont_ts[1:]]
                cur_ref = ref
            elif is_label_only and not cur_label:
                flush_item()
                out.append(f"**{all_txt}**" + (f"  `{ref}`" if ref else ""))
            else:
                cur_body.append(all_txt)
                if ref:
                    cur_ref = (cur_ref + " " + ref).strip() if cur_ref else ref

        flush_item()
        return "\n".join(out) + "\n"

    # ── Upper detail table (防水工事 etc. at Y > table_y_hi) ──────────────
    upper_texts = sorted(
        [t for t in all_texts if t.y > table_y_hi and t.y < title_y - 5],
        key=lambda t: (-t.y, t.x),
    )

    def render_upper() -> str:
        if not upper_texts:
            return ""
        rows = rows_from_items(upper_texts, y_tol=3.0)
        lines: list[str] = ["\n## 防水工事　詳細\n"]
        for row in rows:
            parts = " | ".join(t.text.strip() for t in sorted(row, key=lambda t: t.x) if t.text.strip())
            if parts:
                lines.append(f"- {parts}")
        return "\n".join(lines) + "\n"

    # ── Assemble ──────────────────────────────────────────────────────────
    title_info = _extract_title_block(doc, msp)
    md = f"# {dxf_path.stem}\n\n"
    md += _title_block_md(title_info)

    # Section label from bottom strip
    strip_title = title_info.get("図面名", "特記仕様書")
    md += f"## {strip_title}\n"
    md += render_upper()

    md += "\n## 工事特記仕様\n"
    # Interleave columns top-to-bottom (sections sorted by y_top desc across cols)
    all_sections: list[tuple[float, int, dict]] = []
    for ci in range(len(col_bounds)):
        for sec in group_sections(ci):
            all_sections.append((sec["y_top"], ci, sec))
    all_sections.sort(key=lambda v: -v[0])

    for _, ci, sec in all_sections:
        lo = col_bounds[ci][0]
        md += render_section(sec, lo)

    return md


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection & dispatch
# ─────────────────────────────────────────────────────────────────────────────

def detect_type(dxf_path: Path) -> str:
    """Return 'd003', 'd004', or 'd005' based on drawing content heuristics."""
    texts = _extract_texts(ezdxf.readfile(str(dxf_path)).modelspace())
    text_set = {t.text for t in texts}
    # D003 has SECTION_TITLES in 工事区分 tables
    if text_set & SECTION_TITLES:
        return "d003"
    # D004 has 工事区分 header + h≈4.5 equipment section headers
    if "工事区分" in text_set and any(abs(t.height - 4.5) < 0.2 for t in texts):
        return "d004"
    # D005 has h=3.5 工事種別 headers (仮設工事, 防水工事, etc.)
    if any(abs(t.height - 3.5) < 0.1 for t in texts):
        return "d005"
    return "d003"


def generate(dxf_path: str | Path, doc_type: str | None = None) -> str:
    """Generate Markdown string for the given DXF file.

    Args:
        dxf_path: Path to the DXF file.
        doc_type:  ``'d003'`` | ``'d004'`` | ``'d005'`` | ``None`` (auto-detect).

    Returns:
        Markdown string.
    """
    p = Path(dxf_path)
    if not p.exists():
        raise FileNotFoundError(p)
    dtype = doc_type or detect_type(p)
    if dtype == "d004":
        return _gen_d004(p)
    if dtype == "d005":
        return _gen_d005(p)
    return _gen_d003(p)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured Markdown from DXF spec sheets (D003 / D004).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dxf", help="Input DXF file")
    parser.add_argument("-o", "--output", help="Output .md file (default: <dxf stem>_grid.md)")
    parser.add_argument(
        "--type", choices=["d003", "d004", "d005"],
        help="Force document type (default: auto-detect)",
    )
    args = parser.parse_args()

    dxf_path = Path(args.dxf)
    if not dxf_path.exists():
        print(f"Error: file not found: {dxf_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else dxf_path.with_name(dxf_path.stem + "_grid.md")

    print(f"Reading : {dxf_path}")
    md = generate(dxf_path, doc_type=args.type)
    out_path.write_text(md, encoding="utf-8")
    print(f"Written : {out_path}  ({len(md):,} chars)")


if __name__ == "__main__":
    main()
