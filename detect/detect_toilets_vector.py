"""
Vector-based toilet (便器) detector for CAD PDF drawings.

Mục tiêu:
  Phát hiện bồn cầu từ vector path trong PDF (không cần LLM, không render image để detect).

Heuristic chính:
  1) Bowl outline là stroked path có kích thước gần vuông ~20-24pt.
  2) Path có số đoạn line trung bình (11-13 items).
  3) Bên trong bowl có ít nhất 1 "inner mark" nhỏ (nút/chi tiết trong lòng bồn).

Cách chạy:
  python detect_toilets_vector.py <file.pdf> [page_index]
"""

import re
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass

import fitz  # pymupdf


# ── Reference constants @ scale 1/100 ─────────────────────────────────────────
# Toilet signature: Glyph đứng = 2 stem dọc + 1 bridge ngang; ngang = xoay 90°.
_STEM_V_W_MIN = 4.8
_STEM_V_W_MAX = 6.4
_STEM_V_H_MIN = 12.2
_STEM_V_H_MAX = 14.2

_STEM_H_W_MIN = 12.2
_STEM_H_W_MAX = 14.2
_STEM_H_H_MIN = 4.8
_STEM_H_H_MAX = 6.4

STEM_NL_MIN = 8   # số line items tối thiểu (scale-independent)
STEM_NL_MAX = 12  # số line items tối đa

_BRIDGE_H_W_MIN = 9.5
_BRIDGE_H_W_MAX = 11.2
_BRIDGE_H_H_MAX = 0.7
BRIDGE_H_NL_MAX = 3  # scale-independent

_BRIDGE_V_W_MAX = 0.7
_BRIDGE_V_H_MIN = 9.5
_BRIDGE_V_H_MAX = 11.2
BRIDGE_V_NL_MAX = 3  # scale-independent

_STEM_SEP_MIN = 4.8
_STEM_SEP_MAX = 8.5
_STEM_ALIGN_MAX = 2.0

_BRIDGE_MATCH_ALONG_MAX = 2.5
_BRIDGE_MATCH_CROSS_MAX = 8.5

_BBOX_PAD = 2.0
_BBOX_V_EXTRA_TOP    = 9.0
_BBOX_V_EXTRA_BOTTOM = 10.0
_BBOX_V_EXTRA_LEFT   = 1.5
_BBOX_V_EXTRA_RIGHT  = 2.0
_BBOX_H_EXTRA_LEFT   = 9.0
_BBOX_H_EXTRA_RIGHT  = 8.0
_BBOX_H_EXTRA_TOP    = 2.5
_BBOX_H_EXTRA_BOTTOM = 4.5


# ── Scale detection ───────────────────────────────────────────────────────────
def detect_scale(page: fitz.Page) -> float:
    """Đọc 'S = 1/NNN' từ text để suy ra scale. Default 100."""
    for block in page.get_text("blocks"):
        m = re.search(r"S\s*=\s*1\s*/\s*(\d+)", block[4])
        if m:
            return float(m.group(1))
    return 100.0


@dataclass
class ToiletSymbol:
    bbox: tuple  # (x0,y0,x1,y1) in page points
    center: tuple
    score: float
    page: int
    source: str = "unknown"

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "type": "toilet",
            "page": self.page,
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "score": round(self.score, 3),
            "source": self.source,
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        }


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return inter / min(area_a, area_b)


def _union_bbox(a: tuple, b: tuple, pad: float = 0.0) -> tuple:
    return (
        min(a[0], b[0]) - pad,
        min(a[1], b[1]) - pad,
        max(a[2], b[2]) + pad,
        max(a[3], b[3]) + pad,
    )


def find_toilets(page: fitz.Page, page_idx: int) -> list[ToiletSymbol]:
    scale = detect_scale(page)
    f     = scale / 100.0

    # ── Scale thresholds ──────────────────────────────────────────────────────
    stem_v_w_min = _STEM_V_W_MIN * f;  stem_v_w_max = _STEM_V_W_MAX * f
    stem_v_h_min = _STEM_V_H_MIN * f;  stem_v_h_max = _STEM_V_H_MAX * f
    stem_h_w_min = _STEM_H_W_MIN * f;  stem_h_w_max = _STEM_H_W_MAX * f
    stem_h_h_min = _STEM_H_H_MIN * f;  stem_h_h_max = _STEM_H_H_MAX * f
    bridge_h_w_min = _BRIDGE_H_W_MIN * f;  bridge_h_w_max = _BRIDGE_H_W_MAX * f
    bridge_h_h_max = _BRIDGE_H_H_MAX * f
    bridge_v_w_max = _BRIDGE_V_W_MAX * f
    bridge_v_h_min = _BRIDGE_V_H_MIN * f;  bridge_v_h_max = _BRIDGE_V_H_MAX * f
    stem_sep_min   = _STEM_SEP_MIN   * f;  stem_sep_max   = _STEM_SEP_MAX   * f
    stem_align_max = _STEM_ALIGN_MAX * f
    bridge_along   = _BRIDGE_MATCH_ALONG_MAX * f
    bridge_cross   = _BRIDGE_MATCH_CROSS_MAX * f
    bbox_pad       = _BBOX_PAD       * f
    v_extra_top    = _BBOX_V_EXTRA_TOP    * f;  v_extra_bottom = _BBOX_V_EXTRA_BOTTOM * f
    v_extra_left   = _BBOX_V_EXTRA_LEFT   * f;  v_extra_right  = _BBOX_V_EXTRA_RIGHT  * f
    h_extra_left   = _BBOX_H_EXTRA_LEFT   * f;  h_extra_right  = _BBOX_H_EXTRA_RIGHT  * f
    h_extra_top    = _BBOX_H_EXTRA_TOP    * f;  h_extra_bottom = _BBOX_H_EXTRA_BOTTOM * f
    same_x_tol     = 10.0 * f
    same_y_tol     = 60.0 * f
    ref_sep        = 5.8  * f

    drawings = page.get_drawings()
    cands: list[ToiletSymbol] = []
    stems_v:   list[dict] = []
    stems_h:   list[dict] = []
    bridges_h: list[dict] = []
    bridges_v: list[dict] = []

    # ── Collect primitive components ──────────────────────────────────────────
    for d in drawings:
        rr  = d["rect"]
        bw, bh = rr.width, rr.height
        items = d.get("items", [])
        nl  = sum(1 for it in items if it[0] == "l")
        cx  = (rr.x0 + rr.x1) / 2
        cy  = (rr.y0 + rr.y1) / 2
        entry = {"rect": rr, "cx": cx, "cy": cy, "nl": nl}

        if stem_v_w_min <= bw <= stem_v_w_max and stem_v_h_min <= bh <= stem_v_h_max and STEM_NL_MIN <= nl <= STEM_NL_MAX:
            stems_v.append(entry)
        if stem_h_w_min <= bw <= stem_h_w_max and stem_h_h_min <= bh <= stem_h_h_max and STEM_NL_MIN <= nl <= STEM_NL_MAX:
            stems_h.append(entry)
        if bridge_h_w_min <= bw <= bridge_h_w_max and bh <= bridge_h_h_max and nl <= BRIDGE_H_NL_MAX:
            bridges_h.append(entry)
        if bw <= bridge_v_w_max and bridge_v_h_min <= bh <= bridge_v_h_max and nl <= BRIDGE_V_NL_MAX:
            bridges_v.append(entry)

    # ── Pattern 1: vertical glyph ─────────────────────────────────────────────
    for i, a in enumerate(stems_v):
        for j, b in enumerate(stems_v):
            if j <= i:
                continue
            if abs(a["cy"] - b["cy"]) > stem_align_max:
                continue
            sep = abs(a["cx"] - b["cx"])
            if not (stem_sep_min <= sep <= stem_sep_max):
                continue

            mid_x = (a["cx"] + b["cx"]) / 2
            mid_y = (a["cy"] + b["cy"]) / 2
            bridge = None
            for br in bridges_h:
                if abs(br["cx"] - mid_x) <= bridge_along and abs(br["cy"] - mid_y) <= bridge_cross:
                    bridge = br
                    break
            if bridge is None:
                continue

            bbox = _union_bbox(
                (a["rect"].x0, a["rect"].y0, a["rect"].x1, a["rect"].y1),
                (b["rect"].x0, b["rect"].y0, b["rect"].x1, b["rect"].y1),
                pad=bbox_pad,
            )
            bbox = _union_bbox(bbox, (bridge["rect"].x0, bridge["rect"].y0,
                                      bridge["rect"].x1, bridge["rect"].y1), pad=0)
            bbox = (
                bbox[0] - v_extra_left,
                bbox[1] - v_extra_top,
                bbox[2] + v_extra_right,
                bbox[3] + v_extra_bottom,
            )
            score = 3.0 - 0.15 * abs(sep - ref_sep) - 0.10 * abs(a["cy"] - b["cy"])
            cands.append(ToiletSymbol(
                bbox=bbox,
                center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                score=score,
                page=page_idx,
                source="glyph_vertical",
            ))

    # ── Pattern 2: horizontal glyph (rotated 90°) ────────────────────────────
    for i, a in enumerate(stems_h):
        for j, b in enumerate(stems_h):
            if j <= i:
                continue
            if abs(a["cx"] - b["cx"]) > stem_align_max:
                continue
            sep = abs(a["cy"] - b["cy"])
            if not (stem_sep_min <= sep <= stem_sep_max):
                continue

            mid_x = (a["cx"] + b["cx"]) / 2
            mid_y = (a["cy"] + b["cy"]) / 2
            bridge = None
            for br in bridges_v:
                if abs(br["cy"] - mid_y) <= bridge_along and abs(br["cx"] - mid_x) <= bridge_cross:
                    bridge = br
                    break
            if bridge is None:
                continue

            bbox = _union_bbox(
                (a["rect"].x0, a["rect"].y0, a["rect"].x1, a["rect"].y1),
                (b["rect"].x0, b["rect"].y0, b["rect"].x1, b["rect"].y1),
                pad=bbox_pad,
            )
            bbox = _union_bbox(bbox, (bridge["rect"].x0, bridge["rect"].y0,
                                      bridge["rect"].x1, bridge["rect"].y1), pad=0)
            bbox = (
                bbox[0] - h_extra_left,
                bbox[1] - h_extra_top,
                bbox[2] + h_extra_right,
                bbox[3] + h_extra_bottom,
            )
            score = 3.0 - 0.15 * abs(sep - ref_sep) - 0.10 * abs(a["cx"] - b["cx"])
            cands.append(ToiletSymbol(
                bbox=bbox,
                center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                score=score,
                page=page_idx,
                source="glyph_horizontal",
            ))

    # ── Dedup ─────────────────────────────────────────────────────────────────
    def _same_fixture(a: ToiletSymbol, b: ToiletSymbol) -> bool:
        return (abs(a.center[0] - b.center[0]) <= same_x_tol and
                abs(a.center[1] - b.center[1]) <= same_y_tol)

    cands.sort(key=lambda t: -t.score)
    out: list[ToiletSymbol] = []
    for c in cands:
        if any(_bbox_iou(c.bbox, e.bbox) > 0.6 for e in out):
            continue
        if any(_same_fixture(c, e) for e in out):
            continue
        out.append(c)
    return out


def draw_toilets(page: fitz.Page, toilets: list[ToiletSymbol], out_path: str) -> None:
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)

    from PIL import Image, ImageDraw
    import io

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    scale = 3.0

    for i, t in enumerate(toilets):
        x0, y0, x1, y1 = [v * scale for v in t.bbox]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 170, 170), width=3)
        draw.text((x0, max(0, y0 - 14)), f"WC {i+1}", fill=(0, 170, 170))

    img.save(out_path)
    print(f"  🖼️  Saved: {out_path}  ({len(toilets)} toilets)")


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf/TLC_BZ商品計画テキスト2013モデルプラン.pdf"
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
        print(f"\n📄 Page {pi} / {total - 1}  ({page.rect.width:.0f}×{page.rect.height:.0f} pt)")
        toilets = find_toilets(page, pi)
        print(f"  → {len(toilets)} toilets")
        if toilets:
            out_png = out_dir / f"{pdf_stem}_page{pi}_toilets_vector.png"
            draw_toilets(page, toilets, str(out_png))
            all_results[pi] = [t.to_dict() for t in toilets]

    out_json = out_dir / f"{pdf_stem}_toilets_vector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
