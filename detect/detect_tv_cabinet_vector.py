"""
Vector detector for TV cabinet symbols (class 3).

Ground truth seed:
  rendered_TLC_BZ商品計画テキスト2013モデルプラン_dpi300/page_3.txt

Run:
  python detect_tv_cabinet_vector.py <file.pdf> [page_index]
"""

import re
import sys
import json
import statistics
from pathlib import Path
from dataclasses import dataclass

import fitz  # pymupdf


# ── Reference constants @ scale 1/100 ────────────────────────────────────────
_FILLED_BAR_W_MAX  = 12.0   # pt — max width of a single filled bar
_FILLED_BAR_H_MAX  = 46.0   # pt — max height of a single filled bar
_FILLED_BAR_MIN    = 0.5    # pt — min side length
_RAIL_THIN_MAX     = 1.2    # pt — rail thickness
_RAIL_SPAN_MIN     = 28.0   # pt — rail minimum span
_RAIL_SPAN_MAX     = 95.0   # pt — rail maximum span
RAIL_NL_MIN        = 12     # line items (scale-independent)

_CLUSTER_DX        = 10.0   # pt — grow threshold x
_CLUSTER_DY        = 16.0   # pt — grow threshold y
_CORE_DX           = 15.0   # pt — refined core half-width
_CORE_DY           = 42.0   # pt — refined core half-height
CLUSTER_MIN        = 10     # count (scale-independent)

_SPAN_V_X_MIN      = 4.0;  _SPAN_V_X_MAX = 22.0
_SPAN_V_Y_MIN      = 30.0; _SPAN_V_Y_MAX = 80.0
_SPAN_H_X_MIN      = 30.0; _SPAN_H_X_MAX = 80.0
_SPAN_H_Y_MIN      = 4.0;  _SPAN_H_Y_MAX = 22.0

_RAIL_CHECK_DX     = 14.0   # pt — max rail distance from cluster center (x)
_RAIL_CHECK_DY     = 42.0   # pt — max rail distance from cluster center (y)
RAIL_MIN_COUNT     = 4      # scale-independent (≥4 để loại FP có đúng 3 rails)

_BBOX_HALF_NARROW  = 6.5    # pt — half-width on narrow axis
_BBOX_HALF_WIDE    = 22.0   # pt — half-width on wide axis


def detect_scale(page: fitz.Page) -> float:
    """Đọc 'S = 1/NNN' từ text để suy ra scale. Default 100."""
    for block in page.get_text("blocks"):
        m = re.search(r"S\s*=\s*1\s*/\s*(\d+)", block[4])
        if m:
            return float(m.group(1))
    return 100.0


@dataclass
class TVCabinetSymbol:
    bbox: tuple
    center: tuple
    score: float
    page: int
    orientation: str

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "type": "tv_cabinet",
            "page": self.page,
            "orientation": self.orientation,
            "score": round(self.score, 3),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        }


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return inter / min(area_a, area_b)


def _center(r: fitz.Rect) -> tuple[float, float]:
    return (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2


def find_tv_cabinets(page: fitz.Page, page_idx: int) -> list[TVCabinetSymbol]:
    scale = detect_scale(page)
    f     = scale / 100.0

    # ── Scale thresholds ──────────────────────────────────────────────────────
    bar_w_max    = _FILLED_BAR_W_MAX  * f
    bar_h_max    = _FILLED_BAR_H_MAX  * f
    bar_min      = _FILLED_BAR_MIN    * f
    rail_thin    = _RAIL_THIN_MAX     * f
    rail_span_lo = _RAIL_SPAN_MIN     * f
    rail_span_hi = _RAIL_SPAN_MAX     * f
    clus_dx      = _CLUSTER_DX        * f
    clus_dy      = _CLUSTER_DY        * f
    core_dx      = _CORE_DX           * f
    core_dy      = _CORE_DY           * f
    sv_x_min = _SPAN_V_X_MIN * f;  sv_x_max = _SPAN_V_X_MAX * f
    sv_y_min = _SPAN_V_Y_MIN * f;  sv_y_max = _SPAN_V_Y_MAX * f
    sh_x_min = _SPAN_H_X_MIN * f;  sh_x_max = _SPAN_H_X_MAX * f
    sh_y_min = _SPAN_H_Y_MIN * f;  sh_y_max = _SPAN_H_Y_MAX * f
    rail_dx  = _RAIL_CHECK_DX  * f
    rail_dy  = _RAIL_CHECK_DY  * f
    half_n   = _BBOX_HALF_NARROW * f
    half_w   = _BBOX_HALF_WIDE   * f

    drawings    = page.get_drawings()
    small_filled: list[tuple] = []
    rails:        list[tuple] = []

    for d in drawings:
        r       = d["rect"]
        w, h    = r.width, r.height
        items   = d.get("items", [])
        nl      = sum(1 for it in items if it[0] == "l")
        has_fill   = d.get("fill") is not None
        cx, cy  = _center(r)

        # Small filled bar
        if has_fill and nl <= 4:
            if bar_min <= w <= bar_w_max and bar_min <= h <= bar_h_max:
                small_filled.append((d, cx, cy, w, h))
            elif bar_min <= h <= bar_w_max and bar_min <= w <= bar_h_max:
                small_filled.append((d, cx, cy, w, h))

        # Thin segmented rails: nhiều line items, rất mỏng theo 1 chiều.
        # PDF này dùng zero-width paths (w=0 hoặc h=0) → không check w>0/h>0.
        if nl >= RAIL_NL_MIN:
            if (w <= rail_thin and rail_span_lo <= h <= rail_span_hi) or \
               (h <= rail_thin and rail_span_lo <= w <= rail_span_hi):
                rails.append((d, cx, cy, w, h))

    out: list[TVCabinetSymbol] = []

    # ── Cluster nearby filled fragments ──────────────────────────────────────
    visited = [False] * len(small_filled)
    for i in range(len(small_filled)):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        grown = True
        while grown:
            grown = False
            for j in range(len(small_filled)):
                if visited[j]:
                    continue
                _, sx, sy, _, _ = small_filled[j]
                if any(abs(sx - small_filled[k][1]) <= clus_dx and
                       abs(sy - small_filled[k][2]) <= clus_dy for k in cluster):
                    visited[j] = True
                    cluster.append(j)
                    grown = True

        if len(cluster) < CLUSTER_MIN:
            continue

        points = [(small_filled[k][1], small_filled[k][2]) for k in cluster]
        med_x = statistics.median(p[0] for p in points)
        med_y = statistics.median(p[1] for p in points)

        # Keep dense core around median — removes inflating annotation fragments
        refined = [
            k for k in cluster
            if abs(small_filled[k][1] - med_x) <= core_dx
            and abs(small_filled[k][2] - med_y) <= core_dy
        ]
        if len(refined) < CLUSTER_MIN:
            continue

        pts2 = [(small_filled[k][1], small_filled[k][2]) for k in refined]
        xs = [p[0] for p in pts2]
        ys = [p[1] for p in pts2]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        is_vertical   = sv_x_min <= span_x <= sv_x_max and sv_y_min <= span_y <= sv_y_max
        is_horizontal = sh_x_min <= span_x <= sh_x_max and sh_y_min <= span_y <= sh_y_max
        if not (is_vertical or is_horizontal):
            continue

        # Verify nearby rails
        rail_count = 0
        for _, rx, ry, rw, rh in rails:
            if abs(rx - cx) > rail_dx or abs(ry - cy) > rail_dy:
                continue
            if is_vertical and rh >= rw:
                rail_count += 1
            elif is_horizontal and rw >= rh:
                rail_count += 1
        if rail_count < RAIL_MIN_COUNT:
            continue

        orientation = "vertical" if is_vertical else "horizontal"
        if orientation == "vertical":
            bbox = (cx - half_n, cy - half_w, cx + half_n, cy + half_w)
        else:
            bbox = (cx - half_w, cy - half_n, cx + half_w, cy + half_n)

        score = 3.0 + min(len(refined), 20) * 0.08 + min(rail_count, 8) * 0.05
        out.append(TVCabinetSymbol(
            bbox=bbox,
            center=(cx, cy),
            score=score,
            page=page_idx,
            orientation=orientation,
        ))

    out.sort(key=lambda s: -s.score)
    dedup: list[TVCabinetSymbol] = []
    for s in out:
        if any(_bbox_iou(s.bbox, e.bbox) > 0.5 for e in dedup):
            continue
        dedup.append(s)
    return dedup


def draw_tv_cabinets(page: fitz.Page, symbols: list[TVCabinetSymbol], out_path: str) -> None:
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)

    from PIL import Image, ImageDraw
    import io

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(symbols):
        x0, y0, x1, y1 = [v * 3.0 for v in s.bbox]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 128, 255), width=3)
        draw.text((x0, max(0, y0 - 14)), f"TV{i+1}", fill=(0, 128, 255))
    img.save(out_path)
    print(f"  🖼️  Saved: {out_path}  ({len(symbols)} tv cabinets)")


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf/TLC_BZ商品計画テキスト2013モデルプラン.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc = fitz.open(pdf_path)
    pages = [page_arg] if page_arg is not None else range(len(doc))
    out_dir = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem

    payload: dict = {}
    for pi in pages:
        page = doc[pi]
        symbols = find_tv_cabinets(page, pi)
        print(f"📄 Page {pi}: {len(symbols)} tv cabinets")
        if symbols:
            out_png = out_dir / f"{stem}_page{pi}_tv_cabinets_vector.png"
            draw_tv_cabinets(page, symbols, str(out_png))
            payload[pi] = [s.to_dict() for s in symbols]

    out_json = out_dir / f"{stem}_tv_cabinets_vector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
