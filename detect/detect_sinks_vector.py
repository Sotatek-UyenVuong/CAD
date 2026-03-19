"""
Vector-based sink detector — topology / geometry approach (line-polygon edition).

Sink signature (observed in PDF):
  • Outer rectangular / square frame  (12–32 pt at S=1/100)
  • Exactly 1 inner polygon-circle    (2–8 pt, ≥8 line items, near-square)
  • If > 1 distinct polygon-circle inside → NOT a sink

Two outer-frame cases:
  Case A — single compound drawing:  one drawing whose bbox covers 12–32 pt,
            near-square, ≥8 line items, contains the inner-circle center.
  Case B — corner-mark frame:        four tiny near-square drawings at the
            corners of an implied rectangle; outer rect estimated from them.

Scale-aware: reads "S=1/xxx" from page text and scales all pt thresholds.

Run:
  python detect_sinks_vector.py <file.pdf> [page_index]
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

# Inner polygon-circle
INNER_MIN_PT  = 2.0      # minimum diameter
INNER_MAX_PT  = 8.0      # maximum diameter
INNER_ASP_MIN = 0.80     # min(w,h)/max(w,h) — must be very round
INNER_SIDES   = 8        # minimum number of line items

# Outer frame (single-drawing case)
OUTER_MIN_PT  = 12.0
OUTER_MAX_PT  = 32.0
OUTER_ASP_MIN = 0.55     # allows slightly rectangular (kitchen sink)

# Corner-mark case: tiny near-square marks at the 4 corners
CORNER_MIN_PT = 0.5
CORNER_MAX_PT = 3.5
CORNER_ASP    = 0.60
CORNER_SEARCH = 20.0     # max distance from circle centre to corner

# Two circles are "the same" (concentric rings) if their centres are ≤ this
CONCENTRIC_DIST = 3.0


# ── data class ────────────────────────────────────────────────────────────────
@dataclass
class SinkDetection:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    page: int
    score: float
    source: str = "topology"

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "type": "sink",
            "page": self.page,
            "source": self.source,
            "score": round(self.score, 3),
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


# ── step 1: find inner polygon-circles ───────────────────────────────────────
def _find_inner_circles(drawings: list, inner_min: float, inner_max: float) -> list:
    """
    A polygon-circle is a drawing whose bounding rect is near-square and small,
    contains only line items, and has enough sides to approximate a circle.
    """
    result = []
    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if not (inner_min <= w <= inner_max and inner_min <= h <= inner_max):
            continue
        if max(w, h) < 0.5:
            continue
        if min(w, h) / max(w, h) < INNER_ASP_MIN:
            continue
        nc, nl, _ = _item_counts(d.get("items", []))
        if nc > 0:           # any bezier → not the polygon type we want
            continue
        if nl < INNER_SIDES:
            continue
        result.append(d)
    return result


# ── step 2: cluster concentric rings into one logical circle ─────────────────
def _cluster_circles(circles: list) -> list[list]:
    """
    Group drawings that represent the same physical circle
    (concentric rings = two paths with nearly identical centres).
    Returns list of groups; each group is a list of drawing dicts.
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


def _group_rect(group: list) -> fitz.Rect:
    """Union bounding rect of all drawings in a group."""
    x0 = min(d["rect"].x0 for d in group)
    y0 = min(d["rect"].y0 for d in group)
    x1 = max(d["rect"].x1 for d in group)
    y1 = max(d["rect"].y1 for d in group)
    return fitz.Rect(x0, y0, x1, y1)


def _group_center(group: list) -> tuple[float, float]:
    r = _group_rect(group)
    return _rect_center(r)


# ── step 3a: outer frame — single compound drawing ───────────────────────────
def _find_single_frame(drawings: list, cx: float, cy: float,
                       outer_min: float, outer_max: float):
    """
    Find the best single drawing that looks like the outer frame and
    contains (cx, cy).  Returns the drawing dict or None.
    """
    best = None
    best_area = float("inf")
    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if not (outer_min <= w <= outer_max and outer_min <= h <= outer_max):
            continue
        asp = min(w, h) / max(w, h)
        if asp < OUTER_ASP_MIN:
            continue
        # Centre of inner circle must be inside this drawing's rect
        if not (r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1):
            continue
        nc, nl, nr = _item_counts(d.get("items", []))
        if nc > 0:
            continue   # bezier curves → not the line-based outer frame
        if nl < 4:
            continue   # must have at least 4 lines (rectangle outline)
        area = w * h
        if area < best_area:
            best_area = area
            best = d
    return best


# ── step 3b: outer frame — four corner marks ─────────────────────────────────
def _find_corner_frame(drawings: list, cx: float, cy: float,
                       search_r: float, outer_min: float, outer_max: float):
    """
    Detect a sink frame from four tiny corner marks (one per quadrant).
    For each of TL / TR / BL / BR, picks the smallest-area corner mark
    so that multiple concentric corner layers don't bloat the bounding box.
    Returns fitz.Rect or None.
    """
    # quadrant key → (area, rect)  — keep only the smallest per quadrant
    quadrants: dict[str, tuple[float, fitz.Rect]] = {}

    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if not (CORNER_MIN_PT <= w <= CORNER_MAX_PT and
                CORNER_MIN_PT <= h <= CORNER_MAX_PT):
            continue
        asp = min(w, h) / max(w, h)
        if asp < CORNER_ASP:
            continue
        dcx, dcy = _rect_center(r)
        if math.hypot(dcx - cx, dcy - cy) > search_r:
            continue
        _, nl, _ = _item_counts(d.get("items", []))
        if nl < 3:              # corners need ≥ 3 line items
            continue
        if nl >= INNER_SIDES:   # polygon-circles excluded
            continue

        # Assign to quadrant based on position relative to inner circle centre
        q = ("T" if dcy <= cy else "B") + ("L" if dcx <= cx else "R")
        area = w * h
        if q not in quadrants or area < quadrants[q][0]:
            quadrants[q] = (area, r)

    # All four quadrants must have a corner mark
    if len(quadrants) < 4:
        return None

    rects = [v[1] for v in quadrants.values()]
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    w, h = x1 - x0, y1 - y0

    if not (outer_min <= w <= outer_max and outer_min <= h <= outer_max):
        return None

    return fitz.Rect(x0, y0, x1, y1)


# ── step 4: count circle groups inside a rect ────────────────────────────────
def _circles_inside(groups: list[list], frame: fitz.Rect,
                    margin: float = 2.0) -> list:
    """Return circle groups whose centre is inside frame (±margin)."""
    inside = []
    for g in groups:
        gcx, gcy = _group_center(g)
        if (frame.x0 - margin <= gcx <= frame.x1 + margin and
                frame.y0 - margin <= gcy <= frame.y1 + margin):
            inside.append(g)
    return inside


# ── main detector ─────────────────────────────────────────────────────────────
def find_sinks(page: fitz.Page, page_idx: int) -> list[SinkDetection]:
    scale = detect_scale(page)
    f     = scale / 100.0          # scale factor vs reference S=1/100

    inner_min  = INNER_MIN_PT  * f
    inner_max  = INNER_MAX_PT  * f
    outer_min  = OUTER_MIN_PT  * f
    outer_max  = OUTER_MAX_PT  * f
    search_r   = CORNER_SEARCH * f

    drawings = page.get_drawings()

    # Step 1-2: polygon circles → groups
    circles = _find_inner_circles(drawings, inner_min, inner_max)
    groups  = _cluster_circles(circles)

    cands: list[SinkDetection] = []

    for group in groups:
        cx, cy = _group_center(group)

        # Step 3a: look for a single compound outer frame
        frame_d = _find_single_frame(drawings, cx, cy, outer_min, outer_max)
        if frame_d is not None:
            frame_rect = frame_d["rect"]
        else:
            # Step 3b: look for a corner-mark frame
            frame_rect = _find_corner_frame(drawings, cx, cy,
                                            search_r, outer_min, outer_max)

        if frame_rect is None:
            continue  # no recognisable outer frame → skip

        # Step 4: exactly 1 circle group inside the frame?
        inner = _circles_inside(groups, frame_rect)
        if len(inner) != 1:
            continue  # 0 or >1 circles → not a sink

        pad = 1.5
        x0, y0 = frame_rect.x0 - pad, frame_rect.y0 - pad
        x1, y1 = frame_rect.x1 + pad, frame_rect.y1 + pad
        source = "single-frame" if frame_d is not None else "corner-frame"
        cands.append(SinkDetection(
            bbox=(x0, y0, x1, y1),
            center=((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            page=page_idx,
            score=5.0 if frame_d is not None else 4.5,
            source=source,
        ))

    # Dedup by IoU > 0.5 (keep highest score)
    cands.sort(key=lambda x: -x.score)
    out: list[SinkDetection] = []
    for c in cands:
        if not any(_bbox_iou(c.bbox, e.bbox) > 0.5 for e in out):
            out.append(c)
    return out


# ── visualisation ─────────────────────────────────────────────────────────────
def draw_sinks(page: fitz.Page, sinks: list[SinkDetection], out_path: str) -> None:
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image, ImageDraw
    import io
    img  = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    scale_px = 3.0
    for i, s in enumerate(sinks):
        x0, y0, x1, y1 = [v * scale_px for v in s.bbox]
        color = (50, 200, 100) if s.source == "single-frame" else (50, 140, 255)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 14)), f"S{i + 1}", fill=color)
    img.save(out_path)
    print(f"  Saved: {out_path}  ({len(sinks)} sinks)")


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
        print(f"\nPage {pi}/{total - 1}  "
              f"({page.rect.width:.0f}x{page.rect.height:.0f} pt)  S=1/{scale:.0f}")
        sinks = find_sinks(page, pi)
        print(f"  -> {len(sinks)} sinks  "
              + ", ".join(f"S{i+1}({s.source})" for i, s in enumerate(sinks)))
        if sinks:
            out_png = out_dir / f"{stem}_page{pi}_sinks_vector.png"
            draw_sinks(page, sinks, str(out_png))
            all_results[pi] = [s.to_dict() for s in sinks]

    out_json = out_dir / f"{stem}_sinks_vector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
