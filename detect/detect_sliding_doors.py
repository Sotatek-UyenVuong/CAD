"""
Sliding door detector for CAD PDF drawings.  v2

Phát hiện ký hiệu cửa trượt (引き戸 / 引き分け戸) từ vector PDF.

Nguyên lý (v2):
  Cửa trượt trong PDF này được vẽ bằng 2 đường song song gồm nhiều
  đoạn nhỏ liên tiếp (simulated dashes), KHÔNG phải dashed rect.
  Pattern:
    • 2 stroked paths mỏng (height < 2.5pt hoặc width < 2.5pt)
    • Mỗi path gồm ≥ 12 line items
    • Span 32–95 pt
    • Hai đường song song cách nhau ≤ 5pt, overlap ≥ 70 %
  Bbox cuối = track line + mở rộng 20pt ra 2 phía vuông góc.

Cách chạy:
  python detect_sliding_doors.py <file.pdf> [page_index]
"""

import re
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass

import fitz  # pymupdf
from detect_doors_vector import _try_arc_from_tail, _drawing_points


# ── Reference constants @ scale 1/100 ─────────────────────────────────────────
SEG_MIN_ITEMS  = 12      # số line segments tối thiểu trong một path (scale-independent)
_SEG_MIN_SPAN   = 32.0   # pt @ 1/100
_SEG_MAX_SPAN   = 95.0   # pt @ 1/100
_SEG_MAX_THICK  = 2.5    # pt @ 1/100
_PAIR_MAX_GAP   = 4.0    # pt @ 1/100
PAIR_MIN_OVERLAP = 0.70  # tỉ lệ overlap (dimensionless)
_BBOX_PAD       = 20.0   # pt @ 1/100

_ARC_R_MIN      = 14.0   # pt @ 1/100
_ARC_R_MAX      = 18.0   # pt @ 1/100
_ARC_PERP_DIST  = 23.0   # pt @ 1/100
_ARC_END_DIST   = 22.5   # pt @ 1/100

_IND_W_MIN      = 9.0    # pt @ 1/100
_IND_W_MAX      = 17.0   # pt @ 1/100
_IND_H_MIN      = 11.0   # pt @ 1/100
_IND_H_MAX      = 17.0   # pt @ 1/100
_IND_SPACING_MIN = 18.0  # pt @ 1/100
_IND_SPACING_MAX = 60.0  # pt @ 1/100
_IND_TRACK_DIST  = 35.0  # pt @ 1/100

ARC_SWEEP_MIN   = 80.0   # degrees (scale-independent)
IND_N_MIN       = 8      # items count (scale-independent)
IND_N_MAX       = 25     # items count (scale-independent)


# ── Scale detection ──────────────────────────────────────────────────────────
def detect_scale(page: fitz.Page) -> float:
    """Đọc 'S = 1/NNN' từ text để suy ra scale. Default 100."""
    for block in page.get_text("blocks"):
        m = re.search(r"S\s*=\s*1\s*/\s*(\d+)", block[4])
        if m:
            return float(m.group(1))
    return 100.0


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class SlidingDoor:
    bbox:     tuple   # (x0, y0, x1, y1)
    orient:   str     # 'H' | 'V'
    span:     float   # độ dài track (pt)
    panels:   int     # 1 = 片引き戸, 2 = 引き分け戸

    def to_dict(self):
        return {
            "type":      "sliding",
            "orient":    self.orient,
            "span":      round(self.span, 1),
            "panels":    self.panels,
            "bbox":      [round(v, 2) for v in self.bbox],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bbox_iou(a: tuple, b: tuple) -> float:
    ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
    ix = max(0.0, min(ax1,bx1) - max(ax0,bx0))
    iy = max(0.0, min(ay1,by1) - max(ay0,by0))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = max((ax1-ax0)*(ay1-ay0), 1e-6)
    area_b = max((bx1-bx0)*(by1-by0), 1e-6)
    return inter / min(area_a, area_b)


# ── Detector ──────────────────────────────────────────────────────────────────
def _collect_arcs(page: fitz.Page, arc_r_min: float, arc_r_max: float) -> list[tuple]:
    """Thu thập các arc nhỏ (1/4 tròn) trên trang."""
    arcs = []
    for d in page.get_drawings():
        r = d['rect']
        w, h = r.width, r.height
        if w < 5 or h < 5:
            continue
        if max(w, h) / max(min(w, h), 0.1) > 2.0:
            continue
        pts = _drawing_points(d)
        if len(pts) < 5:
            continue
        arc = _try_arc_from_tail(pts)
        if arc and arc_r_min <= arc.radius <= arc_r_max and arc.sweep >= ARC_SWEEP_MIN:
            arcs.append((arc.center[0], arc.center[1], arc.radius))
    return arcs


def _has_arc_at_track_end(track_bbox: tuple, orient: str,
                           arcs: list[tuple],
                           arc_perp_dist: float, arc_end_dist: float) -> bool:
    """
    Kiểm tra có arc nằm ngay tại ĐẦU TRACK không.

    H-track: arc phải nằm gần y-midline (≤ arc_perp_dist) VÀ
             gần x-endpoint trái hoặc phải (≤ arc_end_dist).
    V-track: tương tự xoay 90°.
    """
    tx0, ty0, tx1, ty1 = track_bbox
    if orient == 'H':
        cy_mid = (ty0 + ty1) / 2
        for cx, cy, _ in arcs:
            if abs(cy - cy_mid) > arc_perp_dist:
                continue
            if abs(cx - tx0) <= arc_end_dist or abs(cx - tx1) <= arc_end_dist:
                return True
    else:  # V
        cx_mid = (tx0 + tx1) / 2
        for cx, cy, _ in arcs:
            if abs(cx - cx_mid) > arc_perp_dist:
                continue
            if abs(cy - ty0) <= arc_end_dist or abs(cy - ty1) <= arc_end_dist:
                return True
    return False


def _find_indicator_pairs(page: fitz.Page,
                           ind_w_min: float, ind_w_max: float,
                           ind_h_min: float, ind_h_max: float,
                           ind_spacing_min: float, ind_spacing_max: float) -> list[tuple]:
    """
    Tìm các cặp ký hiệu tay cầm cửa trượt (2 path n=8-25, size≈11-17pt).
    Trả về list (orient, ind_cross, track_a, track_b).
    """
    cands = []
    for d in page.get_drawings():
        items = d.get('items', [])
        nl = sum(1 for it in items if it[0] == 'l')
        if not (IND_N_MIN <= nl <= IND_N_MAX):
            continue
        r = d['rect']
        if not (ind_w_min <= r.width <= ind_w_max and ind_h_min <= r.height <= ind_h_max):
            continue
        cands.append(r)

    pairs = []
    used = [False] * len(cands)
    for i, r1 in enumerate(cands):
        if used[i]:
            continue
        for j, r2 in enumerate(cands):
            if j <= i or used[j]:
                continue
            # Cùng y level → H-oriented indicator pair
            if abs(r1.y0 - r2.y0) <= 4 and abs(r1.height - r2.height) <= 3:
                spacing = abs(r1.x0 - r2.x0)
                if ind_spacing_min <= spacing <= ind_spacing_max:
                    a = min(r1.x0, r2.x0)
                    b = max(r1.x1, r2.x1)
                    ind_y = (r1.y0 + r2.y0) / 2
                    pairs.append(('H', ind_y, a, b))
                    used[i] = used[j] = True
                    break
            # Cùng x level → V-oriented indicator pair
            elif abs(r1.x0 - r2.x0) <= 4 and abs(r1.width - r2.width) <= 3:
                spacing = abs(r1.y0 - r2.y0)
                if ind_spacing_min <= spacing <= ind_spacing_max:
                    a = min(r1.y0, r2.y0)
                    b = max(r1.y1, r2.y1)
                    ind_x = (r1.x0 + r2.x0) / 2
                    pairs.append(('V', ind_x, a, b))
                    used[i] = used[j] = True
                    break
    return pairs


def _extract_seg_lines(page: fitz.Page,
                        seg_min_span: float, seg_max_span: float,
                        seg_max_thick: float) -> list:
    """
    Tìm tất cả paths mỏng nhiều đoạn = ký hiệu sliding door track.
    Trả về list của (orient, cross_coord, start, end, span, nl).
    """
    result = []
    for d in page.get_drawings():
        items = d.get('items', [])
        nl = sum(1 for it in items if it[0] == 'l')
        if nl < SEG_MIN_ITEMS:
            continue
        r = d['rect']
        w, h = r.width, r.height
        # Phải rất mỏng theo 1 chiều
        if h > seg_max_thick and w > seg_max_thick:
            continue
        span = max(w, h)
        if not (seg_min_span <= span <= seg_max_span):
            continue
        orient = 'H' if w >= h else 'V'
        if orient == 'H':
            result.append(('H', r.y0, r.x0, r.x1, span, nl))
        else:
            result.append(('V', r.x0, r.y0, r.y1, span, nl))
    return result


def find_sliding_doors(page: fitz.Page) -> list[SlidingDoor]:
    """Detect cửa trượt từ segmented parallel line pairs."""
    scale = detect_scale(page)
    f     = scale / 100.0

    # ── Scale thresholds ──────────────────────────────────────────────────────
    seg_min_span   = _SEG_MIN_SPAN   * f
    seg_max_span   = _SEG_MAX_SPAN   * f
    seg_max_thick  = _SEG_MAX_THICK  * f
    pair_max_gap   = _PAIR_MAX_GAP   * f
    bbox_pad       = _BBOX_PAD       * f
    arc_r_min      = _ARC_R_MIN      * f
    arc_r_max      = _ARC_R_MAX      * f
    arc_perp_dist  = _ARC_PERP_DIST  * f
    arc_end_dist   = _ARC_END_DIST   * f
    ind_w_min      = _IND_W_MIN      * f
    ind_w_max      = _IND_W_MAX      * f
    ind_h_min      = _IND_H_MIN      * f
    ind_h_max      = _IND_H_MAX      * f
    ind_spacing_min = _IND_SPACING_MIN * f
    ind_spacing_max = _IND_SPACING_MAX * f
    ind_track_dist  = _IND_TRACK_DIST  * f
    ind_h_max_pad   = ind_h_max
    ind_w_max_pad   = ind_w_max
    end_pad  = 25.0 * f
    perp_pad = 28.0 * f
    double_gap_max = 12.0 * f

    segs = _extract_seg_lines(page, seg_min_span, seg_max_span, seg_max_thick)
    W_pt = page.rect.width
    H_pt = page.rect.height
    arcs = _collect_arcs(page, arc_r_min, arc_r_max)

    used  = [False] * len(segs)
    doors: list[SlidingDoor] = []

    # ── Bước 1: ghép cặp parallel lines ─────────────────────────────────────
    for i, s1 in enumerate(segs):
        if used[i]:
            continue
        o1, cross1, a1, b1, sp1, n1 = s1
        best_j, best_gap = None, float('inf')

        for j, s2 in enumerate(segs):
            if j == i or used[j]:
                continue
            o2, cross2, a2, b2, sp2, n2 = s2
            if o1 != o2:
                continue
            gap = abs(cross1 - cross2)
            if gap > pair_max_gap:
                continue
            # Primary axis overlap
            overlap = min(b1, b2) - max(a1, a2)
            if overlap < min(sp1, sp2) * PAIR_MIN_OVERLAP:
                continue
            # Span similarity
            if abs(sp1 - sp2) / max(sp1, sp2, 1) > 0.22:
                continue
            if gap < best_gap:
                best_gap, best_j = gap, j

        if best_j is None:
            continue

        s2 = segs[best_j]
        o2, cross2, a2, b2, sp2, n2 = s2
        used[i] = used[best_j] = True

        c_lo = min(cross1, cross2)
        c_hi = max(cross1, cross2)
        a    = max(a1, a2)
        b    = min(b1, b2)
        span = b - a

        if o1 == 'H':
            bbox = (
                max(0,    a - end_pad),
                max(0,    c_lo - perp_pad),
                min(W_pt, b + end_pad),
                min(H_pt, c_hi + perp_pad),
            )
        else:
            bbox = (
                max(0,    c_lo - perp_pad),
                max(0,    a - end_pad),
                min(W_pt, c_hi + perp_pad),
                min(H_pt, b + end_pad),
            )

        # ── Xác nhận: phải có arc indicator tại đầu track ────────────────────
        if o1 == 'H':
            thin_bbox = (max(0, a - 2), max(0, c_lo - 2),
                         min(W_pt, b + 2), min(H_pt, c_hi + 2))
        else:
            thin_bbox = (max(0, c_lo - 2), max(0, a - 2),
                         min(W_pt, c_hi + 2), min(H_pt, b + 2))
        if not _has_arc_at_track_end(thin_bbox, o1, arcs, arc_perp_dist, arc_end_dist):
            continue

        doors.append(SlidingDoor(bbox=bbox, orient=o1, span=span, panels=1))

    # ── Fallback: tìm cửa bằng cặp indicator paths ───────────────────────────
    ind_pairs = _find_indicator_pairs(
        page,
        ind_w_min, ind_w_max,
        ind_h_min, ind_h_max,
        ind_spacing_min, ind_spacing_max,
    )
    for ind_orient, ind_cross, ind_a, ind_b in ind_pairs:
        best_track = None
        best_dist  = float('inf')
        for o, cross, ta, tb, sp, n in segs:
            if o != ind_orient:
                continue
            if n < 20:
                continue
            overlap = min(tb, ind_b) - max(ta, ind_a)
            if overlap < (ind_b - ind_a) * 0.50:
                continue
            dist_val = ind_cross - cross
            if 3 <= dist_val <= ind_track_dist and dist_val < best_dist:
                best_dist, best_track = dist_val, (o, cross, ta, tb, sp, n)

        if best_track is None:
            continue

        o2, cross2, ta2, tb2, sp2, n2 = best_track
        if ind_orient == 'H':
            full = (
                max(0,    min(ta2, ind_a) - end_pad),
                max(0,    cross2 - perp_pad),
                min(W_pt, max(tb2, ind_b) + end_pad),
                min(H_pt, ind_cross + ind_h_max_pad + 4),
            )
        else:
            full = (
                max(0,    ind_cross - ind_w_max_pad - 4),
                max(0,    min(ta2, ind_a) - end_pad),
                min(W_pt, cross2 + perp_pad),
                min(H_pt, max(tb2, ind_b) + end_pad),
            )

        if any(_bbox_iou(full, d.bbox) > 0.4 for d in doors):
            continue

        doors.append(SlidingDoor(bbox=full, orient=ind_orient, span=sp2, panels=1))

    # ── Bước 2: ghép 2 panels cạnh nhau → 引き分け戸 ──────────────────────
    used2 = [False] * len(doors)
    merged: list[SlidingDoor] = []

    for i, d1 in enumerate(doors):
        if used2[i]:
            continue
        best_j2 = None
        best_dist2 = float('inf')

        for j, d2 in enumerate(doors):
            if j == i or used2[j]:
                continue
            if d1.orient != d2.orient:
                continue
            if d1.orient == 'H':
                gap = min(abs(d1.bbox[2] - d2.bbox[0]),
                          abs(d2.bbox[2] - d1.bbox[0]))
                cy1 = (d1.bbox[1] + d1.bbox[3]) / 2
                cy2 = (d2.bbox[1] + d2.bbox[3]) / 2
                if abs(cy1 - cy2) > bbox_pad * 1.5:
                    continue
            else:
                gap = min(abs(d1.bbox[3] - d2.bbox[1]),
                          abs(d2.bbox[3] - d1.bbox[1]))
                cx1 = (d1.bbox[0] + d1.bbox[2]) / 2
                cx2 = (d2.bbox[0] + d2.bbox[2]) / 2
                if abs(cx1 - cx2) > bbox_pad * 1.5:
                    continue
            if gap > double_gap_max:
                continue
            if abs(d1.span - d2.span) / max(d1.span, d2.span, 1) > 0.25:
                continue
            if gap < best_dist2:
                best_dist2, best_j2 = gap, j

        if best_j2 is not None:
            d2 = doors[best_j2]
            used2[i] = used2[best_j2] = True
            combined = (
                min(d1.bbox[0], d2.bbox[0]), min(d1.bbox[1], d2.bbox[1]),
                max(d1.bbox[2], d2.bbox[2]), max(d1.bbox[3], d2.bbox[3]),
            )
            merged.append(SlidingDoor(
                bbox=combined,
                orient=d1.orient,
                span=max(d1.span, d2.span),
                panels=2,
            ))
        else:
            used2[i] = True
            merged.append(d1)

    # ── Dedup: loại bbox overlap > 60 % ──────────────────────────────────────
    deduped: list[SlidingDoor] = []
    for d in merged:
        if not any(_bbox_iou(d.bbox, e.bbox) > 0.6 for e in deduped):
            deduped.append(d)

    return deduped


# ── Visualize ─────────────────────────────────────────────────────────────────
def draw_sliding_doors(page: fitz.Page, doors: list[SlidingDoor], out_path: str):
    """Render trang → PNG và vẽ sliding doors (xanh dương)."""
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)

    from PIL import Image, ImageDraw
    import io

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    scale = 3.0

    for i, s in enumerate(doors):
        x0, y0, x1, y1 = [v * scale for v in s.bbox]
        label = f"slide {i+1}" + (" ++" if s.panels == 2 else "")
        draw.rectangle([x0, y0, x1, y1], outline=(30, 100, 220), width=3)
        draw.text((x0, max(0, y0 - 14)), label, fill=(30, 100, 220))

    img.save(out_path)
    print(f"  🖼️  Saved: {out_path}  ({len(doors)} sliding doors)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else \
        "pdf/TLC_BZ商品計画テキスト2013モデルプラン.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc   = fitz.open(pdf_path)
    total = len(doc)
    pages = [page_arg] if page_arg is not None else range(total)

    out_dir  = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    pdf_stem = Path(pdf_path).stem

    all_results = {}

    for pi in pages:
        page = doc[pi]
        drawings_count = len(page.get_drawings())
        images_count = len(page.get_images(full=True))
        print(f"\n📄 Page {pi} / {total-1}  "
              f"({page.rect.width:.0f}×{page.rect.height:.0f} pt)")

        doors = find_sliding_doors(page)
        print(f"  → {len(doors)} sliding doors  "
              f"({sum(1 for d in doors if d.panels==2)} 引き分け戸)")
        if not doors and images_count > 0 and drawings_count < 200:
            print("  ⚠️  Trang dạng raster/cover (vector ít), tạm thời bỏ qua — sẽ xử lý sau.")

        if doors:
            out_png = out_dir / f"{pdf_stem}_page{pi}_sliding_vector.png"
            draw_sliding_doors(page, doors, str(out_png))
            all_results[pi] = [d.to_dict() for d in doors]

    out_json = out_dir / f"{pdf_stem}_sliding_vector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
