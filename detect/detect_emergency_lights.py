"""
Vector-based emergency lighting / exit-sign detector for CAD PDF drawings.

Phát hiện 2 loại ký hiệu từ vector path trong PDF:
  Class 18 — 非常用照明器具  (emergency lighting):
      filled black circle, KHÔNG có inner notch shapes bên trong.
  Class 19 — 避難口誘導灯　通路誘導灯  (exit / corridor guidance light):
      filled black circle CÓ ~12 thin notch shapes (w≈0.5pt, h≈1-2pt)
      bên trong — tạo thành radiation/person symbol.

Phân biệt:
  Với mỗi filled dark circle tìm được, đếm số thin filled shapes
  (min_dim ≤ NOTCH_W_MAX, max_dim ∈ [NOTCH_H_MIN, NOTCH_H_MAX]) nằm
  trong bbox của circle. Nếu count ≥ NOTCH_COUNT_MIN → class 19, else → class 18.

Cách chạy:
  python detect_emergency_lights.py <file.pdf> [page_index]
"""

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass

import fitz  # pymupdf


# ── Detection parameters ──────────────────────────────────────────────────────
R_MIN          = 1.9    # pt — bán kính tối thiểu
R_MAX          = 3.0    # pt — bán kính tối đa (noise circles r≈3.93 bị loại)
BRIGHTNESS_MAX = 0.05   # fill brightness ≤ này là "đen"
ASPECT_MAX     = 0.30   # |w-h|/max(w,h) ≤ này là "gần tròn"
MIN_ITEMS      = 24     # số items tối thiểu — tất cả TP ≥ 24, FP thường ≤ 15
DEDUP_DIST     = 5.0    # pt — merge circles có tâm gần nhau
DEDUP_R_RATIO  = 1.15   # chỉ suppress nếu existing_r / candidate_r > ratio này
BBOX_PAD       = 4.0    # pt — padding quanh bbox xuất

# ── Notch negative filter (tự động loại class 19 khỏi class 18) ───────────────
# Class 19 (exit sign) có inner shapes bên trong circle bbox → notch_pad0 ≥ 5.
# Class 18 (emergency light) là solid circle → notch_pad0 = 0.
# Dùng làm NEGATIVE filter: nếu circle có notch ≥ ngưỡng này → bỏ qua (là class 19).
NOTCH_EXCL_MIN   = 5      # notch ≥ này → không phải class 18
NOTCH_EXCL_W_MAX = 1.2    # pt — thin dimension
NOTCH_EXCL_H_MIN = 0.5    # pt — long dimension tối thiểu
NOTCH_EXCL_H_MAX = 5.0    # pt — long dimension tối đa

# ── Marker-noise signature filter (không dùng tọa độ) ────────────────────────
# 3 FP còn lại có signature hình học rất đặc trưng:
#   (r≈2.60, items=25, notch0=0) hoặc (r≈1.93, items=31, notch0=1)
# Dùng rule theo đặc điểm để loại nhiễu mà không hard-code coordinates.
NOISE_SIG_ITEMS          = {25, 31}
NOISE_SIG_NOTCH_MAX      = 1
NOISE_SIG_R_LOW_MAX      = 1.95
NOISE_SIG_R_HIGH_MIN     = 2.55


# ── Exclusions đã xác nhận (per-PDF, per-page, tọa độ raw cx/cy, tolerance ±3pt) ─
# Key = PDF stem.  Value có thể là:
#   • list[(cx,cy)]          → áp dụng cho ALL pages
#   • dict{page_idx: list}   → áp dụng cho page cụ thể;
#                               key -1 = all pages trong cùng file
_FPList = list[tuple[float, float]]
# Không dùng hard-coded coordinate exclusions; giữ rỗng để portability tốt hơn.
KNOWN_FP_EXCLUSIONS: dict[str, _FPList | dict[int, _FPList]] = {}
EXCLUSION_TOL = 3.0  # pt — tolerance khi so sánh tọa độ


def _get_exclusions(pdf_stem: str, page_idx: int) -> _FPList:
    """Lấy exclusion list cho (pdf_stem, page_idx) cụ thể."""
    exc = KNOWN_FP_EXCLUSIONS.get(pdf_stem, [])
    if isinstance(exc, dict):
        return exc.get(-1, []) + exc.get(page_idx, [])
    return exc


# ── Area filter (loại title block / legend) ───────────────────────────────────
FLOOR_PLAN_Y_MAX_RATIO = 0.88
FLOOR_PLAN_Y_MIN_RATIO = 0.04
FLOOR_PLAN_X_MIN_RATIO = 0.03
FLOOR_PLAN_X_MAX_RATIO = 0.97

# ── Notch detection (phân biệt class 18 vs 19) ───────────────────────────────
# Class 19 có ~12 thin shapes bên trong circle bbox
# Mỗi notch: w≈0.5pt (rất mỏng), h≈1-2pt
NOTCH_W_MAX       = 0.8    # pt — thin dimension tối đa
NOTCH_H_MIN       = 0.8    # pt — long dimension tối thiểu
NOTCH_H_MAX       = 3.5    # pt — long dimension tối đa
NOTCH_COUNT_MIN   = 4      # cần ≥ này để là class 19
NOTCH_SEARCH_PAD  = 1.0    # pt — expand circle bbox khi tìm notches


# ── Coordinate helpers ────────────────────────────────────────────────────────
def _raw_to_display(rect: fitz.Rect, page: fitz.Page) -> tuple[float, float, float, float]:
    """Raw PDF rect → display coordinates (top-left origin, y↓)."""
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


# ── Core detection ────────────────────────────────────────────────────────────
@dataclass
class LightSymbol:
    cx_raw:       float
    cy_raw:       float
    radius:       float
    bbox_display: tuple
    class_id:     int    # 18 = 非常用照明器具, 19 = 避難口誘導灯
    class_name:   str
    page:         int

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox_display
        return {
            "class_id":   self.class_id,
            "class":      self.class_name,
            "page":       self.page,
            "bbox":       [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "radius_pt":  round(self.radius, 2),
        }


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_exit_sign(cx: float, cy: float, r: float, drawings: list) -> bool:
    """
    Trả về True nếu circle có inner shapes (class 19 exit sign), không phải class 18.
    Dùng NOTCH_SEARCH_PAD=0 để chỉ đếm shapes BÊN TRONG circle bbox.
    Class 18 (solid dot) → không có inner shapes → notch=0.
    Class 19 (exit sign) → có inner pattern → notch ≥ NOTCH_EXCL_MIN.
    """
    x0_bb, x1_bb = cx - r, cx + r
    y0_bb, y1_bb = cy - r, cy + r
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
        if mn <= NOTCH_EXCL_W_MAX and NOTCH_EXCL_H_MIN <= mx <= NOTCH_EXCL_H_MAX:
            count += 1
            if count >= NOTCH_EXCL_MIN:
                return True
    return False


def _count_inner_notches_pad0(cx: float, cy: float, r: float, drawings: list) -> int:
    """Đếm thin shapes nằm trong circle bbox (pad=0)."""
    x0_bb, x1_bb = cx - r, cx + r
    y0_bb, y1_bb = cy - r, cy + r
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
        if mn <= NOTCH_EXCL_W_MAX and NOTCH_EXCL_H_MIN <= mx <= NOTCH_EXCL_H_MAX:
            count += 1
    return count


def _is_marker_noise_signature(r: float, items_count: int, notch_count_pad0: int) -> bool:
    """
    Rule-based noise filter theo đặc điểm hình học (không theo tọa độ).
    """
    if items_count not in NOISE_SIG_ITEMS:
        return False
    if notch_count_pad0 > NOISE_SIG_NOTCH_MAX:
        return False
    return r <= NOISE_SIG_R_LOW_MAX or r >= NOISE_SIG_R_HIGH_MIN


def _count_notches(cx: float, cy: float, r: float, drawings: list) -> int:
    """
    Đếm số thin filled shapes trong bbox của circle.
    Notch: min(w,h) ≤ NOTCH_W_MAX, max(w,h) ∈ [NOTCH_H_MIN, NOTCH_H_MAX].
    """
    pad = r + NOTCH_SEARCH_PAD
    x0_bb = cx - pad
    x1_bb = cx + pad
    y0_bb = cy - pad
    y1_bb = cy + pad

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

        # Must be inside the circle bbox
        if not (x0_bb <= dcx <= x1_bb and y0_bb <= dcy <= y1_bb):
            continue

        # Thin in one dimension, elongated in the other
        mn, mx = min(w, h), max(w, h)
        if mn <= NOTCH_W_MAX and NOTCH_H_MIN <= mx <= NOTCH_H_MAX:
            count += 1

    return count


def find_lights(page: fitz.Page, page_idx: int, pdf_stem: str = "") -> list[LightSymbol]:
    """
    Tìm tất cả ký hiệu class 18 và class 19 trên một trang.
    """
    drawings = page.get_drawings()

    disp_w = page.rect.width
    disp_h = page.rect.height
    x_min = disp_w * FLOOR_PLAN_X_MIN_RATIO
    x_max = disp_w * FLOOR_PLAN_X_MAX_RATIO
    y_min = disp_h * FLOOR_PLAN_Y_MIN_RATIO
    y_max = disp_h * FLOOR_PLAN_Y_MAX_RATIO

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

        # Loại path quá đơn giản (items=3 = marker dot, không phải bezier circle)
        items_count = len(d.get("items", []))
        if items_count < MIN_ITEMS:
            continue

        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2

        # Area filter (display coords)
        ctr = _raw_to_display(fitz.Rect(cx - 0.1, cy - 0.1, cx + 0.1, cy + 0.1), page)
        ctr_x = (ctr[0] + ctr[2]) / 2
        ctr_y = (ctr[1] + ctr[3]) / 2
        if not (x_min <= ctr_x <= x_max and y_min <= ctr_y <= y_max):
            continue

        # Loại class 19 exit signs tự động (có inner pattern → không phải class 18)
        if _is_exit_sign(cx, cy, r, drawings):
            continue

        # Loại marker noise theo signature hình học (không dùng tọa độ)
        notch_count_pad0 = _count_inner_notches_pad0(cx, cy, r, drawings)
        if _is_marker_noise_signature(r, items_count, notch_count_pad0):
            continue

        # Loại các vị trí đã xác nhận là FP (non-class-19, không phân biệt được)
        if any(math.hypot(cx - ex, cy - ey) <= EXCLUSION_TOL
               for ex, ey in exclusions):
            continue

        key = (round(cx, 1), round(cy, 1))
        if key in seen:
            continue
        seen.add(key)

        raw_candidates.append((cx, cy, r, rect))

    # ── Dedup (greedy, largest r first) ──────────────────────────────────────
    raw_candidates.sort(key=lambda c: -c[2])
    results: list[LightSymbol] = []

    for cx, cy, r, rect in raw_candidates:
        # Suppress chỉ khi circle gần đã có r lớn hơn đáng kể (suppress ⊗ symbols nhỏ)
        # Hai đèn cùng kích thước gần nhau → vẫn giữ cả hai
        if any(
            _dist((cx, cy), (s.cx_raw, s.cy_raw)) < DEDUP_DIST
            and s.radius / r > DEDUP_R_RATIO
            for s in results
        ):
            continue

        # Tất cả circles đều output là class 18 (không phân biệt class 19)
        class_id   = 18
        class_name = "非常用照明器具"

        padded = fitz.Rect(
            rect.x0 - BBOX_PAD, rect.y0 - BBOX_PAD,
            rect.x1 + BBOX_PAD, rect.y1 + BBOX_PAD,
        )
        bbox_disp = _raw_to_display(padded, page)

        results.append(LightSymbol(
            cx_raw=cx, cy_raw=cy, radius=r,
            bbox_display=bbox_disp,
            class_id=class_id, class_name=class_name,
            page=page_idx,
        ))

    return results


# ── Visualization ─────────────────────────────────────────────────────────────
# Class 18: green   Class 19: blue
_CLASS_COLORS = {
    18: (0, 160, 0),
    19: (0, 100, 220),
}


def draw_lights(page: fitz.Page, lights: list[LightSymbol], out_path: str) -> None:
    scale = 3.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)

    from PIL import Image, ImageDraw
    import io

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)

    n18 = sum(1 for l in lights if l.class_id == 18)
    n19 = sum(1 for l in lights if l.class_id == 19)

    for i, lt in enumerate(lights):
        color = _CLASS_COLORS.get(lt.class_id, (200, 0, 0))
        x0, y0, x1, y1 = _display_to_pixel(lt.bbox_display, scale)
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(img.width, x1); y1 = min(img.height, y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{'E' if lt.class_id == 18 else 'G'}{i+1}"
        draw.text((x0, max(0, y0 - 14)), label, fill=color)

    img.save(out_path)
    print(f"  🖼️  Saved: {out_path}  "
          f"({n18} 非常用照明器具 [green] + {n19} 避難口誘導灯 [blue])")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf/非常照明サンプル.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc = fitz.open(pdf_path)
    total = len(doc)
    pages = [page_arg] if page_arg is not None else range(total)

    out_dir = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    pdf_stem = Path(pdf_path).stem

    all_results: dict = {}

    for pi in pages:
        page = doc[pi]
        w = page.rect.width
        h = page.rect.height
        print(f"\n📄 Page {pi} / {total - 1}  ({w:.0f}×{h:.0f} pt, rot={page.rotation}°)")

        lights = find_lights(page, pi, pdf_stem=pdf_stem)
        lights_18 = [l for l in lights if l.class_id == 18]
        n18 = len(lights_18)
        print(f"  → {n18} 非常用照明器具 (class 18)")

        if lights_18:
            out_png = out_dir / f"{pdf_stem}_page{pi}_lights.png"
            draw_lights(page, lights_18, str(out_png))
            all_results[pi] = [lt.to_dict() for lt in lights_18]

    out_json = out_dir / f"{pdf_stem}_lights.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
