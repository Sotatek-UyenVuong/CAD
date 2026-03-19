"""
Vector-based exit / corridor guidance light detector for CAD PDF drawings.

Class 19 — 避難口誘導灯　通路誘導灯 (exit / corridor guidance light):
    Filled black circle CÓ thin notch shapes bên trong
    (radiation / person symbol pattern).

Cách chạy:
  python detect_exit_signs.py <file.pdf> [page_index]
"""

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass

import fitz  # pymupdf


# ── Detection parameters ──────────────────────────────────────────────────────
R_MIN          = 1.9    # pt — bán kính tối thiểu
R_MAX          = 3.0    # pt — bán kính tối đa (noise circles có r≈3.93 bị loại)
BRIGHTNESS_MAX = 0.05   # fill brightness ≤ này là "đen"
ASPECT_MAX     = 0.30   # |w-h|/max(w,h) ≤ này là "gần tròn"
MIN_ITEMS      = 8      # items tối thiểu; notch check bù cho filter này
DEDUP_DIST     = 5.0    # pt — merge circles có tâm gần nhau
DEDUP_R_RATIO  = 1.15   # chỉ suppress nếu existing_r / candidate_r > ratio
BBOX_PAD       = 4.0    # pt — padding quanh bbox xuất

# ── Notch detection parameters ────────────────────────────────────────────────
# Class 19 có ~12 thin filled shapes NẰM TRONG circle bbox.
# Class 18 (solid black dot) không có inner shapes → notch=0 khi pad=0.
# Insight: FP circles chỉ có notch từ shapes *bên ngoài* → khi pad=0 notch<5.
#          TP circles có inner pattern thật → khi pad=0 notch≥5–13.
NOTCH_W_MAX      = 1.2    # pt — thin dimension tối đa
NOTCH_H_MIN      = 0.5    # pt — long dimension tối thiểu
NOTCH_H_MAX      = 5.0    # pt — long dimension tối đa
NOTCH_COUNT_MIN  = 5      # cần ≥ này để phân loại là class 19
NOTCH_SEARCH_PAD = 0.0    # pt — chỉ tìm trong circle bbox, KHÔNG expand ra ngoài

# ── Area filter (loại title block, margin) ────────────────────────────────────
FLOOR_PLAN_Y_MIN_RATIO = 0.04
FLOOR_PLAN_Y_MAX_RATIO = 0.88
FLOOR_PLAN_X_MIN_RATIO = 0.03
FLOOR_PLAN_X_MAX_RATIO = 0.97

EXCLUSION_TOL = 3.0  # pt — tolerance so sánh tọa độ exclusion

# ── Known FP exclusions (per-PDF, per-page) ───────────────────────────────────
_FPList = list[tuple[float, float]]
KNOWN_FP_EXCLUSIONS: dict[str, _FPList | dict[int, _FPList]] = {
    # Thêm exclusion khi cần:
    # "pdf_stem": [(cx, cy), ...],
    # "pdf_stem": {0: [(cx,cy),...], 1: [(cx,cy),...], -1: [(cx,cy),...]},
}


def _get_exclusions(pdf_stem: str, page_idx: int) -> _FPList:
    exc = KNOWN_FP_EXCLUSIONS.get(pdf_stem, [])
    if isinstance(exc, dict):
        return exc.get(-1, []) + exc.get(page_idx, [])
    return exc


# ── Coordinate helpers ────────────────────────────────────────────────────────
def _raw_to_display(rect: fitz.Rect, page: fitz.Page) -> tuple[float, float, float, float]:
    rotation = page.rotation
    mbox = page.mediabox
    mW, mH = mbox.width, mbox.height
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

    if rotation == 0:
        return (x0, mH - y1, x1, mH - y0)
    elif rotation == 90:
        return (mH - y1, x0, mH - y0, x1)
    elif rotation == 180:
        return (mW - x1, y0, mW - x0, y1)
    elif rotation == 270:
        return (y0, mW - x1, y1, mW - x0)
    else:
        return (x0, y0, x1, y1)


def _display_to_pixel(bbox: tuple, scale: float) -> tuple:
    return tuple(v * scale for v in bbox)


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── Notch counter ─────────────────────────────────────────────────────────────
def _count_notches(cx: float, cy: float, r: float, drawings: list) -> int:
    """
    Đếm số thin filled shapes nằm trong bbox của circle.
    Notch: min(w,h) ≤ NOTCH_W_MAX  và  max(w,h) ∈ [NOTCH_H_MIN, NOTCH_H_MAX].
    """
    pad   = r + NOTCH_SEARCH_PAD
    x0_bb = cx - pad;  x1_bb = cx + pad
    y0_bb = cy - pad;  y1_bb = cy + pad

    count = 0
    for d in drawings:
        fill = d.get("fill")
        if fill is None or not isinstance(fill, (tuple, list)):
            continue
        if sum(fill[:3]) / 3 > BRIGHTNESS_MAX:
            continue
        dr = d["rect"]
        w, h = dr.width, dr.height
        if w <= 0 or h <= 0:
            continue
        dcx = (dr.x0 + dr.x1) / 2
        dcy = (dr.y0 + dr.y1) / 2
        if not (x0_bb <= dcx <= x1_bb and y0_bb <= dcy <= y1_bb):
            continue
        mn, mx = min(w, h), max(w, h)
        if mn <= NOTCH_W_MAX and NOTCH_H_MIN <= mx <= NOTCH_H_MAX:
            count += 1
    return count


# ── Core detection ────────────────────────────────────────────────────────────
@dataclass
class ExitSignSymbol:
    cx_raw:       float
    cy_raw:       float
    radius:       float
    notch_count:  int
    bbox_display: tuple
    page:         int

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox_display
        return {
            "class_id":    19,
            "class":       "避難口誘導灯　通路誘導灯",
            "page":        self.page,
            "bbox":        [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "radius_pt":   round(self.radius, 2),
            "notch_count": self.notch_count,
        }


def find_exit_signs(page: fitz.Page, page_idx: int, pdf_stem: str = "") -> list[ExitSignSymbol]:
    drawings  = page.get_drawings()
    disp_w    = page.rect.width
    disp_h    = page.rect.height
    x_min     = disp_w * FLOOR_PLAN_X_MIN_RATIO
    x_max     = disp_w * FLOOR_PLAN_X_MAX_RATIO
    y_min     = disp_h * FLOOR_PLAN_Y_MIN_RATIO
    y_max     = disp_h * FLOOR_PLAN_Y_MAX_RATIO
    exclusions = _get_exclusions(pdf_stem, page_idx)

    raw_candidates: list[tuple] = []
    seen: set = set()

    for d in drawings:
        fill = d.get("fill")
        if fill is None or not isinstance(fill, (tuple, list)):
            continue
        if sum(fill[:3]) / 3 > BRIGHTNESS_MAX:
            continue

        rect = d["rect"]
        w, h = rect.width, rect.height
        if w <= 0 or h <= 0:
            continue
        if abs(w - h) / max(w, h) > ASPECT_MAX:
            continue

        r = (w + h) / 4
        if not (R_MIN <= r <= R_MAX):
            continue

        if len(d.get("items", [])) < MIN_ITEMS:
            continue

        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2

        # Area filter
        ctr = _raw_to_display(fitz.Rect(cx - 0.1, cy - 0.1, cx + 0.1, cy + 0.1), page)
        ctr_x = (ctr[0] + ctr[2]) / 2
        ctr_y = (ctr[1] + ctr[3]) / 2
        if not (x_min <= ctr_x <= x_max and y_min <= ctr_y <= y_max):
            continue

        if any(math.hypot(cx - ex, cy - ey) <= EXCLUSION_TOL for ex, ey in exclusions):
            continue

        key = (round(cx, 1), round(cy, 1))
        if key in seen:
            continue
        seen.add(key)

        raw_candidates.append((cx, cy, r, rect))

    # ── Notch filter TRƯỚC dedup ──────────────────────────────────────────────
    # Quan trọng: dedup phải xảy ra SAU khi filter notch, để tránh trường hợp
    # một circle lớn (notch thấp) suppress một circle nhỏ hơn (notch cao) cạnh nó.
    notch_candidates: list[tuple] = []
    for cx, cy, r, rect in raw_candidates:
        notch_count = _count_notches(cx, cy, r, drawings)
        if notch_count >= NOTCH_COUNT_MIN:
            notch_candidates.append((cx, cy, r, rect, notch_count))

    # Dedup chỉ trên những circles đã pass notch filter
    notch_candidates.sort(key=lambda c: -c[2])
    results: list[ExitSignSymbol] = []
    for cx, cy, r, rect, notch_count in notch_candidates:
        if any(
            _dist((cx, cy), (s.cx_raw, s.cy_raw)) < DEDUP_DIST
            and s.radius / r > DEDUP_R_RATIO
            for s in results
        ):
            continue

        padded   = fitz.Rect(rect.x0 - BBOX_PAD, rect.y0 - BBOX_PAD,
                             rect.x1 + BBOX_PAD, rect.y1 + BBOX_PAD)
        bbox_disp = _raw_to_display(padded, page)

        results.append(ExitSignSymbol(
            cx_raw=cx, cy_raw=cy, radius=r,
            notch_count=notch_count,
            bbox_display=bbox_disp,
            page=page_idx,
        ))

    return results


# ── Visualization ─────────────────────────────────────────────────────────────
def draw_exit_signs(page: fitz.Page, signs: list[ExitSignSymbol], out_path: str) -> None:
    scale = 3.0
    mat   = fitz.Matrix(scale, scale)
    pix   = page.get_pixmap(matrix=mat)

    from PIL import Image, ImageDraw
    import io

    img  = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    color = (0, 100, 220)  # blue

    for i, s in enumerate(signs):
        x0, y0, x1, y1 = _display_to_pixel(s.bbox_display, scale)
        x0 = max(0, x0);      y0 = max(0, y0)
        x1 = min(img.width, x1);  y1 = min(img.height, y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 14)), f"G{i+1}(n={s.notch_count})", fill=color)

    img.save(out_path)
    print(f"  🖼️  Saved: {out_path}  ({len(signs)} 避難口誘導灯 [blue])")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf/非常照明サンプル.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc       = fitz.open(pdf_path)
    total     = len(doc)
    pages     = [page_arg] if page_arg is not None else range(total)
    out_dir   = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    pdf_stem  = Path(pdf_path).stem

    all_results: dict = {}

    for pi in pages:
        page = doc[pi]
        w, h = page.rect.width, page.rect.height
        print(f"\n📄 Page {pi} / {total - 1}  ({w:.0f}×{h:.0f} pt, rot={page.rotation}°)")

        signs = find_exit_signs(page, pi, pdf_stem=pdf_stem)
        print(f"  → {len(signs)} 避難口誘導灯 (class 19)  "
              f"[notch range: {min((s.notch_count for s in signs), default=0)}–"
              f"{max((s.notch_count for s in signs), default=0)}]")

        if signs:
            out_png = out_dir / f"{pdf_stem}_page{pi}_exit_signs.png"
            draw_exit_signs(page, signs, str(out_png))
        all_results[pi] = [s.to_dict() for s in signs]

    out_json = out_dir / f"{pdf_stem}_exit_signs.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
