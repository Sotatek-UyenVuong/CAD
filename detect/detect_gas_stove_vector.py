"""
Vector-based gas stove detector — topology / geometry approach.

Gas stove signature (observed in PDF):
  • Outer rectangular / square frame  (12–24 pt at S=1/100)
  • Exactly 3 inner burners, each burner = 2 concentric polygon-circles
    → 6 polygon-circles total in 3 concentric pairs

A polygon-circle is a near-square drawing built from line segments (nl ≥ 8),
as used in this CAD PDF style (no bezier curves).

Scale-aware: reads "S=1/xxx" from page text and scales all pt thresholds.

Run:
  python detect_gas_stove_vector.py <file.pdf> [page_index]
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


# ── reference thresholds at S=1/100, A4 (595×842 pt) ─────────────────────────

# Burner polygon-circle (outer or inner ring of each burner)
BURNER_MIN_PT  = 2.0      # pt — smallest burner ring diameter
BURNER_MAX_PT  = 10.0     # pt — largest burner ring diameter
BURNER_ASP_MIN = 0.75     # min(w,h)/max(w,h) — must be near-square
BURNER_SIDES   = 8        # minimum line items to count as polygon-circle

# Two circles are "concentric" (same burner) if centres ≤ this apart
CONCENTRIC_DIST = 3.0     # pt

# Three burners belong to the same stove if all centres ≤ this radius
BURNER_CLUSTER_R = 12.0   # pt

# Outer stove frame (single compound drawing)
FRAME_MIN_PT  = 11.0
FRAME_MAX_PT  = 26.0
FRAME_ASP_MIN = 0.60      # allows slightly non-square stoves
FRAME_SIDES   = 6         # minimum line items in the outer frame


# ── data class ────────────────────────────────────────────────────────────────
@dataclass
class GasStoveDetection:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    page: int
    score: float
    n_burners: int = 3

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "type": "gas_stove",
            "page": self.page,
            "score": round(self.score, 3),
            "n_burners": self.n_burners,
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        }


# ── small utilities ───────────────────────────────────────────────────────────
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


def _item_counts(items: list) -> tuple[int, int, int]:
    """Return (n_curves, n_lines, n_rects)."""
    nc = sum(1 for it in items if it[0] == "c")
    nl = sum(1 for it in items if it[0] == "l")
    nr = sum(1 for it in items if it[0] == "re")
    return nc, nl, nr


def _rect_center(r: fitz.Rect) -> tuple[float, float]:
    return (r.x0 + r.x1) * 0.5, (r.y0 + r.y1) * 0.5


def detect_scale(page: fitz.Page) -> float:
    """Read drawing scale from page text: 'S=1/100' → 100.0. Default 100."""
    for block in page.get_text("blocks"):
        m = re.search(r"S\s*=\s*1\s*/\s*(\d+)", block[4])
        if m:
            return float(m.group(1))
    return 100.0


# ── step 1: find burner polygon-circles ───────────────────────────────────────
def _find_burner_circles(drawings: list, min_sz: float, max_sz: float) -> list:
    """
    Polygon-circles: near-square, many line items, no bezier curves.
    Each burner is drawn as two concentric polygon-circles.
    """
    result = []
    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if not (min_sz <= w <= max_sz and min_sz <= h <= max_sz):
            continue
        if max(w, h) < 0.5:
            continue
        if min(w, h) / max(w, h) < BURNER_ASP_MIN:
            continue
        nc, nl, _ = _item_counts(d.get("items", []))
        if nc > 0:           # bezier curves → skip
            continue
        if nl < BURNER_SIDES:
            continue
        result.append(d)
    return result


# ── step 2: cluster concentric rings → one burner group ───────────────────────
def _cluster_concentric(circles: list) -> list[list]:
    """
    Group drawings that represent the same burner
    (outer ring + inner ring have nearly identical centres).
    """
    centers = [_rect_center(d["rect"]) for d in circles]
    assigned = [False] * len(circles)
    groups: list[list] = []
    for i in range(len(circles)):
        if assigned[i]:
            continue
        group = [circles[i]]
        assigned[i] = True
        for j in range(i + 1, len(circles)):
            if assigned[j]:
                continue
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if math.hypot(dx, dy) <= CONCENTRIC_DIST:
                group.append(circles[j])
                assigned[j] = True
        groups.append(group)
    return groups


def _group_center(group: list) -> tuple[float, float]:
    xs = [_rect_center(d["rect"])[0] for d in group]
    ys = [_rect_center(d["rect"])[1] for d in group]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ── step 3: cluster 3 burner groups → stove cluster ──────────────────────────
def _find_burner_clusters(groups: list, cluster_r: float) -> list[list[int]]:
    """
    Find sets of ≥ 3 burner groups whose centres all lie within cluster_r
    of each other's centroid.  Returns list of index-sets.
    """
    centers = [_group_center(g) for g in groups]
    used = [False] * len(groups)
    stove_clusters: list[list[int]] = []

    for i, (cx, cy) in enumerate(centers):
        if used[i]:
            continue
        nearby = [j for j, (gx, gy) in enumerate(centers)
                  if math.hypot(gx - cx, gy - cy) <= cluster_r]
        if len(nearby) < 3:
            continue
        for j in nearby:
            used[j] = True
        stove_clusters.append(nearby)

    return stove_clusters


# ── step 4: find outer frame containing all burner centres ────────────────────
def _find_outer_frame(drawings: list, burner_centers: list[tuple],
                      frame_min: float, frame_max: float):
    """
    Return the tightest single drawing that:
      • Is within frame_min–frame_max in both dimensions
      • Is near-square (asp ≥ FRAME_ASP_MIN)
      • Has only line items (no bezier curves), nl ≥ FRAME_SIDES
      • Contains ALL burner centres inside its rect
    """
    best = None
    best_area = float("inf")
    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if not (frame_min <= w <= frame_max and frame_min <= h <= frame_max):
            continue
        asp = min(w, h) / max(w, h)
        if asp < FRAME_ASP_MIN:
            continue
        nc, nl, _ = _item_counts(d.get("items", []))
        if nc > 0:
            continue
        if nl < FRAME_SIDES:
            continue
        # All burner centres must be inside the frame rect
        if not all(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1
                   for cx, cy in burner_centers):
            continue
        area = w * h
        if area < best_area:
            best_area = area
            best = d
    return best


# ── main detector ─────────────────────────────────────────────────────────────
def find_gas_stoves(page: fitz.Page, page_idx: int) -> list[GasStoveDetection]:
    scale  = detect_scale(page)
    f      = scale / 100.0

    burner_min   = BURNER_MIN_PT   * f
    burner_max   = BURNER_MAX_PT   * f
    conc_dist    = CONCENTRIC_DIST * f
    cluster_r    = BURNER_CLUSTER_R * f
    frame_min    = FRAME_MIN_PT    * f
    frame_max    = FRAME_MAX_PT    * f

    drawings = page.get_drawings()

    # Step 1–2: polygon circles → concentric burner groups
    circles = _find_burner_circles(drawings, burner_min, burner_max)
    # Override concentric distance with scaled value
    centers  = [_rect_center(d["rect"]) for d in circles]
    assigned = [False] * len(circles)
    groups: list[list] = []
    for i in range(len(circles)):
        if assigned[i]:
            continue
        group = [circles[i]]
        assigned[i] = True
        for j in range(i + 1, len(circles)):
            if assigned[j]:
                continue
            if math.hypot(centers[i][0] - centers[j][0],
                          centers[i][1] - centers[j][1]) <= conc_dist:
                group.append(circles[j])
                assigned[j] = True
        groups.append(group)

    if len(groups) < 3:
        return []

    # Step 3: find clusters of ≥ 3 burner groups
    stove_clusters = _find_burner_clusters(groups, cluster_r)

    cands: list[GasStoveDetection] = []

    for cluster_indices in stove_clusters:
        burner_groups = [groups[i] for i in cluster_indices]
        burner_centers = [_group_center(g) for g in burner_groups]

        # Step 4: find the outer frame
        frame_d = _find_outer_frame(drawings, burner_centers, frame_min, frame_max)
        if frame_d is None:
            continue

        # Verify: count how many of our burner groups are inside the frame
        fr = frame_d["rect"]
        inside = [c for c in burner_centers
                  if fr.x0 <= c[0] <= fr.x1 and fr.y0 <= c[1] <= fr.y1]
        if len(inside) < 3:
            continue

        pad = 1.5
        x0, y0 = fr.x0 - pad, fr.y0 - pad
        x1, y1 = fr.x1 + pad, fr.y1 + pad
        cands.append(GasStoveDetection(
            bbox=(x0, y0, x1, y1),
            center=((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            page=page_idx,
            score=5.0 + 0.1 * len(inside),
            n_burners=len(inside),
        ))

    # Dedup by IoU > 0.5
    cands.sort(key=lambda x: -x.score)
    out: list[GasStoveDetection] = []
    for c in cands:
        if not any(_bbox_iou(c.bbox, e.bbox) > 0.5 for e in out):
            out.append(c)
    return out


# ── visualisation ─────────────────────────────────────────────────────────────
def draw_gas_stoves(page: fitz.Page, stoves: list[GasStoveDetection],
                    out_path: str) -> None:
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image, ImageDraw
    import io
    img  = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    scale_px = 3.0
    for i, s in enumerate(stoves):
        x0, y0, x1, y1 = [v * scale_px for v in s.bbox]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 90, 0), width=3)
        draw.text((x0, max(0, y0 - 14)), f"G{i+1}({s.n_burners}b)", fill=(255, 90, 0))
    img.save(out_path)
    print(f"  Saved: {out_path}  ({len(stoves)} gas stoves)")


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else \
        "pdf/TLC_BZ商品計画テキスト2013モデルプラン.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc   = fitz.open(pdf_path)
    total = len(doc)
    pages = [page_arg] if page_arg is not None else range(total)

    out_dir = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem
    all_results: dict = {}

    for pi in pages:
        page  = doc[pi]
        scale = detect_scale(page)
        print(f"\nPage {pi}/{total-1}  "
              f"({page.rect.width:.0f}x{page.rect.height:.0f} pt)  S=1/{scale:.0f}")
        stoves = find_gas_stoves(page, pi)
        print(f"  -> {len(stoves)} gas stoves  "
              + ", ".join(f"G{i+1}({s.n_burners}b)" for i, s in enumerate(stoves)))
        if stoves:
            out_png = out_dir / f"{stem}_page{pi}_gas_stoves_vector.png"
            draw_gas_stoves(page, stoves, str(out_png))
            all_results[pi] = [s.to_dict() for s in stoves]

    out_json = out_dir / f"{stem}_gas_stoves_vector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
