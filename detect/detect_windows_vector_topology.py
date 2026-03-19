"""
Pure vector window detector (topology-based, no label coordinates).

Design constraints:
  - Uses only PDF vector line geometry from page.get_drawings()
  - No dependency on external YOLO / txt coordinates
  - Supports:
      * window_vertical
      * window_horizontal
      * fix_window_horizontal (fixed window is horizontal only)

Run:
  python detect_windows_vector_topology.py <file.pdf> [page_index]
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


# Geometry tolerances (PDF pt units)
MIN_SEG_LEN = 0.8
AXIS_EPS = 0.2

# Vertical window topology
V_LONG_MIN = 18.0
V_LONG_MAX = 60.0
V_GAP_MIN = 5.0
V_GAP_MAX = 12.0
V_OVERLAP_MIN = 0.6
V_MID_H_Y_TOL = 2.5
V_MID_H_LEN_RATIO = 0.7
V_INNER_SHORT_MIN_RATIO = 0.22
V_INNER_SHORT_MAX_RATIO = 0.75

# Horizontal window / fixed window topology
H_LONG_MIN = 18.0
H_LONG_MAX = 140.0
H_GAP_MIN = 3.0
H_GAP_MAX = 12.0
H_OVERLAP_MIN = 0.40
H_MAJOR_FIX_MIN = 50.0
H_MAJOR_FIX_MAX = 140.0
H_MAJOR_WINDOW_MAX = 54.0
H_STACK_NEAR_Y = 14.0
# Recall-first mode: allow denser horizontal stacks to avoid misses.
# (User will manually filter noise afterward.)
H_STACK_MAX_LINES = 4
SQUARE_MIN_SIZE = 0.8
SQUARE_MAX_SIZE = 3.0


@dataclass(frozen=True)
class LineSeg:
    p0: tuple[float, float]
    p1: tuple[float, float]
    orientation: str  # "H", "V", "D"
    length: float

    @property
    def x0(self) -> float:
        return min(self.p0[0], self.p1[0])

    @property
    def x1(self) -> float:
        return max(self.p0[0], self.p1[0])

    @property
    def y0(self) -> float:
        return min(self.p0[1], self.p1[1])

    @property
    def y1(self) -> float:
        return max(self.p0[1], self.p1[1])

    @property
    def cx(self) -> float:
        return (self.p0[0] + self.p1[0]) * 0.5

    @property
    def cy(self) -> float:
        return (self.p0[1] + self.p1[1]) * 0.5


@dataclass
class WindowDetection:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    page: int
    orientation: str
    subtype: str  # "window" or "fix_window"
    score: float

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "type": "window",
            "subtype": self.subtype,
            "page": self.page,
            "orientation": self.orientation,
            "score": round(self.score, 3),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        }


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    inter = max(0.0, hi - lo)
    if inter <= 0:
        return 0.0
    den = min(max(a1 - a0, 1e-6), max(b1 - b0, 1e-6))
    return inter / den


def _line_from_item(item) -> LineSeg | None:
    if item[0] != "l":
        return None
    p0 = (item[1].x, item[1].y)
    p1 = (item[2].x, item[2].y)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < MIN_SEG_LEN:
        return None
    if abs(dx) <= abs(dy) * AXIS_EPS:
        ori = "V"
    elif abs(dy) <= abs(dx) * AXIS_EPS:
        ori = "H"
    else:
        ori = "D"
    return LineSeg(p0=p0, p1=p1, orientation=ori, length=length)


def extract_axis_lines(page: fitz.Page) -> tuple[list[LineSeg], list[LineSeg], list[LineSeg]]:
    h: list[LineSeg] = []
    v: list[LineSeg] = []
    d: list[LineSeg] = []
    for drawing in page.get_drawings():
        if "s" not in str(drawing.get("type", "")):
            continue
        for item in drawing.get("items", []):
            seg = _line_from_item(item)
            if seg is None:
                continue
            if seg.orientation == "H":
                h.append(seg)
            elif seg.orientation == "V":
                v.append(seg)
            else:
                d.append(seg)
    return h, v, d


def extract_small_squares(page: fitz.Page) -> list[fitz.Rect]:
    """Extract tiny square markers used at fixed-window endpoints."""
    out: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            t = item[0]
            rr = None
            if t == "re":
                rr = fitz.Rect(item[1].x0, item[1].y0, item[1].x1, item[1].y1)
            elif t == "qu":
                q = item[1]
                xs = [q.ul.x, q.ur.x, q.ll.x, q.lr.x]
                ys = [q.ul.y, q.ur.y, q.ll.y, q.lr.y]
                rr = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            if rr is None:
                continue
            if not (SQUARE_MIN_SIZE <= rr.width <= SQUARE_MAX_SIZE):
                continue
            if not (SQUARE_MIN_SIZE <= rr.height <= SQUARE_MAX_SIZE):
                continue
            out.append(rr)
    return out


def _find_vertical_windows(v_lines: list[LineSeg], h_lines: list[LineSeg], page_idx: int) -> list[WindowDetection]:
    out: list[WindowDetection] = []
    long_vs = [s for s in v_lines if V_LONG_MIN <= s.length <= V_LONG_MAX]
    for i in range(len(long_vs)):
        base_left = long_vs[i]
        for j in range(i + 1, len(long_vs)):
            left = base_left
            right = long_vs[j]
            if right.cx < left.cx:
                left, right = right, left

            gap = right.cx - left.cx
            if not (V_GAP_MIN <= gap <= V_GAP_MAX):
                continue

            ov = _overlap_ratio(left.y0, left.y1, right.y0, right.y1)
            if ov < V_OVERLAP_MIN:
                continue

            y0 = max(left.y0, right.y0)
            y1 = min(left.y1, right.y1)
            h_major = y1 - y0
            if h_major <= 0:
                continue
            y_mid = (y0 + y1) * 0.5

            # One horizontal bridge near center, spanning most of inner gap
            bridge = None
            for h in h_lines:
                if abs(h.cy - y_mid) > V_MID_H_Y_TOL:
                    continue
                if h.x0 > left.cx + 1.5 or h.x1 < right.cx - 1.5:
                    continue
                if h.length < gap * V_MID_H_LEN_RATIO:
                    continue
                bridge = h
                break
            if bridge is None:
                continue

            # Inner short verticals (observed pattern: 2 short vertical lines inside)
            short_min = h_major * V_INNER_SHORT_MIN_RATIO
            short_max = h_major * V_INNER_SHORT_MAX_RATIO
            inner_short_x: list[float] = []
            for v in v_lines:
                if v.length < short_min or v.length > short_max:
                    continue
                if not (left.cx - 0.8 <= v.cx <= right.cx + 0.8):
                    continue
                if not (y0 - 1.5 <= v.cy <= y1 + 1.5):
                    continue
                inner_short_x.append(v.cx)

            inner_short_x.sort()
            inner_short_clusters = []
            for x in inner_short_x:
                if not inner_short_clusters or abs(x - inner_short_clusters[-1]) > 1.1:
                    inner_short_clusters.append(x)
            if len(inner_short_clusters) < 2:
                continue

            x0 = min(left.x0, right.x0, bridge.x0) - 1.0
            x1 = max(left.x1, right.x1, bridge.x1) + 1.0
            y0b = min(left.y0, right.y0) - 1.0
            y1b = max(left.y1, right.y1) + 1.0
            score = 5.0 + min(1.0, len(inner_short_clusters) * 0.2)

            out.append(WindowDetection(
                bbox=(x0, y0b, x1, y1b),
                center=((x0 + x1) * 0.5, (y0b + y1b) * 0.5),
                page=page_idx,
                orientation="vertical",
                subtype="window",
                score=score,
            ))
    return out


def _find_horizontal_windows(
    h_lines: list[LineSeg],
    v_lines: list[LineSeg],
    small_squares: list[fitz.Rect],
    page_idx: int,
) -> list[WindowDetection]:
    out: list[WindowDetection] = []
    long_hs = [s for s in h_lines if H_LONG_MIN <= s.length <= H_LONG_MAX]
    for i in range(len(long_hs)):
        base_top = long_hs[i]
        for j in range(i + 1, len(long_hs)):
            top = base_top
            bot = long_hs[j]
            if bot.cy < top.cy:
                top, bot = bot, top

            gap = bot.cy - top.cy
            if not (H_GAP_MIN <= gap <= H_GAP_MAX):
                continue

            ov = _overlap_ratio(top.x0, top.x1, bot.x0, bot.x1)
            if ov < H_OVERLAP_MIN:
                continue
            # Avoid pairing unrelated rails with very different lengths.
            if min(top.length, bot.length) / max(top.length, bot.length) < 0.40:
                continue

            x0 = max(top.x0, bot.x0)
            x1 = min(top.x1, bot.x1)
            major_len = x1 - x0
            if major_len <= 0:
                continue

            # Reject clutter stacks: many parallel horizontal rails in nearby y
            # with similar x span are usually non-window symbols.
            # Exclude lines much longer than the candidate pair — they belong to
            # a wider adjacent feature (e.g. balcony rail) and should not count.
            stack_count = 0
            for h2 in h_lines:
                if abs(h2.cy - ((top.cy + bot.cy) * 0.5)) > H_STACK_NEAR_Y:
                    continue
                ov2 = _overlap_ratio(x0, x1, h2.x0, h2.x1)
                if ov2 >= 0.75 and h2.length >= major_len * 0.7 and h2.length <= major_len * 1.3:
                    stack_count += 1
            if stack_count > H_STACK_MAX_LINES:
                continue

            # Perpendicular middle vertical support for horizontal window.
            # Keep it slightly relaxed because pairing top/bottom rails can shift candidate center.
            y_mid = (top.cy + bot.cy) * 0.5
            mid_vs = [
                v for v in v_lines
                if (x0 - 1.0) <= v.cx <= (x1 + 1.0)
                and abs(v.cy - y_mid) <= (gap * 1.4)
                and v.length >= max(2.0, gap * 0.55)
            ]
            has_mid_v = len(mid_vs) >= 1

            accepted_fix = False
            if H_MAJOR_FIX_MIN <= major_len <= H_MAJOR_FIX_MAX:
                # Fixed-window strict template:
                # 2 long H (top/bottom) + 1 middle H + 2 short H + 2 endpoint squares.
                inside_h = [
                    h2 for h2 in h_lines
                    if (x0 - 1.0) <= h2.cx <= (x1 + 1.0)
                    and (top.cy + 0.35) <= h2.cy <= (bot.cy - 0.35)
                ]
                mid_min = major_len * 0.35
                mid_max = major_len * 0.80
                short_min = major_len * 0.12
                short_max = major_len * 0.40

                middle_h = [h2 for h2 in inside_h if mid_min <= h2.length <= mid_max]
                short_h = [h2 for h2 in inside_h if short_min <= h2.length <= short_max]

                # short parallels clustered by y (need at least 2 groups)
                short_y = sorted(h2.cy for h2 in short_h)
                short_clusters: list[float] = []
                for yy in short_y:
                    if not short_clusters or abs(yy - short_clusters[-1]) > 1.1:
                        short_clusters.append(yy)

                y_mid = (top.cy + bot.cy) * 0.5
                sq_centers = [
                    ((sq.x0 + sq.x1) * 0.5, (sq.y0 + sq.y1) * 0.5)
                    for sq in small_squares
                    if (x0 + 0.10 * major_len) <= ((sq.x0 + sq.x1) * 0.5) <= (x1 - 0.10 * major_len)
                    and abs(((sq.y0 + sq.y1) * 0.5) - y_mid) <= 3.0
                ]
                has_left_sq = any(cx < (x0 + x1) * 0.5 for cx, _ in sq_centers)
                has_right_sq = any(cx > (x0 + x1) * 0.5 for cx, _ in sq_centers)
                has_center_sq = any(abs(cx - (x0 + x1) * 0.5) <= major_len * 0.12 for cx, _ in sq_centers)

                fix_template_ok = len(middle_h) >= 1
                # Strong inner structure: two mid-length inner lines + two vertical supports.
                # Only relevant for ASYMMETRIC pairs (shared long rail, e.g. page-26 adjacent
                # windows). For symmetric pairs the normal fix-window template is enough;
                # using strong_inner there mis-classifies normal windows as fix_window.
                pair_len_ratio = min(top.length, bot.length) / max(top.length, bot.length)
                is_asymmetric_pair = pair_len_ratio < 0.65
                strong_inner = is_asymmetric_pair and len(middle_h) >= 2 and len(mid_vs) >= 2
                # Some fixed-window variants can lose short-band lines after clipping.
                # Allow only for strong center-marker or strong inner structure variants.
                short_ok = len(short_clusters) >= 1
                if not short_ok:
                    short_ok = (len(middle_h) >= 2 and has_center_sq) or strong_inner
                if not short_ok:
                    fix_template_ok = False
                # Accept four fixed-window marker variants:
                #  1) endpoint markers (left + right)
                #  2) single center marker
                #  3) single endpoint marker (left or right) with >= 2 inner short bands
                #  4) strong inner structure (2 mid-H + 2 mid-V), no sq required
                sq_ok = (has_left_sq and has_right_sq) or has_center_sq
                if not sq_ok:
                    sq_ok = (has_left_sq or has_right_sq) and len(short_clusters) >= 2
                if not sq_ok:
                    sq_ok = strong_inner
                if not fix_template_ok:
                    accepted_fix = False
                elif not sq_ok:
                    accepted_fix = False
                else:
                    # Reject balcony-rail-like pattern:
                    # a one-sided long continuation at the same rail y-level
                    # is typically not a fixed window.
                    left_tail_max = 0.0
                    right_tail_max = 0.0
                    for h2 in h_lines:
                        if abs(h2.cy - top.cy) > 0.35 and abs(h2.cy - bot.cy) > 0.35:
                            continue
                        if (x1 - 1.0) <= h2.x0 <= (x1 + 18.0):
                            right_tail_max = max(right_tail_max, h2.length)
                        if (x0 - 18.0) <= h2.x1 <= (x0 + 1.0):
                            left_tail_max = max(left_tail_max, h2.length)
                    if (
                        major_len <= 75.0
                        and len(sq_centers) >= 2
                        and max(left_tail_max, right_tail_max) >= 45.0
                        and min(left_tail_max, right_tail_max) <= 8.0
                    ):
                        accepted_fix = False
                    else:
                        subtype = "fix_window"  # fixed window horizontal-only
                        score = 5.4
                        accepted_fix = True

            if not accepted_fix:
                if major_len > H_MAJOR_WINDOW_MAX:
                    continue
                if not has_mid_v:
                    continue

                # Enforce template for normal horizontal window:
                # 2 long parallel + >=2 short parallels inside + 1 middle perpendicular.
                short_min = major_len * V_INNER_SHORT_MIN_RATIO
                short_max = major_len * V_INNER_SHORT_MAX_RATIO
                inner_short_y: list[float] = []
                for h2 in h_lines:
                    if h2.length < short_min or h2.length > short_max:
                        continue
                    if not (x0 - 1.0 <= h2.cx <= x1 + 1.0):
                        continue
                    if not (top.cy - 1.5 <= h2.cy <= bot.cy + 1.5):
                        continue
                    inner_short_y.append(h2.cy)
                inner_short_y.sort()
                inner_short_clusters = []
                for y in inner_short_y:
                    if not inner_short_clusters or abs(y - inner_short_clusters[-1]) > 1.1:
                        inner_short_clusters.append(y)
                # Horizontal window variants:
                #  - common: 2 short inner bands
                #  - clipped rare case: 0 short inner bands for near-max width windows.
                # Still reject clutter variants like page-3 W4 (3 bands).
                if len(inner_short_clusters) == 2:
                    pass
                elif len(inner_short_clusters) == 0:
                    if not (35.0 <= major_len <= H_MAJOR_WINDOW_MAX and gap <= 5.2 and stack_count <= 2):
                        continue
                    # Reject corner-junction FPs: mid-V lines must be spread across
                    # at least 20% of the major width, not all clustered at one x.
                    mid_v_xs = [v.cx for v in mid_vs]
                    mid_v_span = (max(mid_v_xs) - min(mid_v_xs)) if len(mid_v_xs) >= 2 else 0.0
                    if mid_v_span < major_len * 0.20:
                        continue
                else:
                    continue

                subtype = "window"
                score = 5.0

            y0b = top.y0 - 1.0
            y1b = bot.y1 + 1.0
            # Use the inner overlap region (x0/x1) for the bounding box so that
            # asymmetric shared-rail pairs (GT1/GT2 sharing a long bottom rail)
            # produce tight boxes instead of spanning both windows.
            x0b = x0 - 1.0
            x1b = x1 + 1.0

            # New fixed-window variant: two adjacent modules merged into one long rail pair.
            # Split into two detections when there is a clear center divider + two inner markers.
            split_boxes: list[tuple[float, float, float, float]] = [(x0b, y0b, x1b, y1b)]
            if subtype == "fix_window":
                center_dividers = [
                    vv for vv in mid_vs
                    if abs(vv.cx - ((x0 + x1) * 0.5)) <= max(1.2, major_len * 0.04)
                ]
                sq_x = sorted(cx for cx, _ in sq_centers)
                if len(center_dividers) >= 1 and len(sq_x) >= 2 and major_len > 90.0:
                    left_rel = (sq_x[0] - x0) / max(major_len, 1e-6)
                    right_rel = (sq_x[-1] - x0) / max(major_len, 1e-6)
                    if 0.18 <= left_rel <= 0.35 and 0.65 <= right_rel <= 0.82:
                        split_x = sum(vv.cx for vv in center_dividers) / len(center_dividers)
                        left_major = split_x - x0
                        right_major = x1 - split_x
                        if left_major >= 22.0 and right_major >= 22.0:
                            split_boxes = [
                                (x0b, y0b, split_x, y1b),
                                (split_x, y0b, x1b, y1b),
                            ]

            for bx0, by0, bx1, by1 in split_boxes:
                out.append(WindowDetection(
                    bbox=(bx0, by0, bx1, by1),
                    center=((bx0 + bx1) * 0.5, (by0 + by1) * 0.5),
                    page=page_idx,
                    orientation="horizontal",
                    subtype=subtype,
                    score=score,
                ))
    return out


def find_windows_topology(page: fitz.Page, page_idx: int) -> list[WindowDetection]:
    h_lines, v_lines, _ = extract_axis_lines(page)
    small_squares = extract_small_squares(page)
    candidates = _find_vertical_windows(v_lines, h_lines, page_idx) + _find_horizontal_windows(
        h_lines, v_lines, small_squares, page_idx
    )

    # Suppress fix-window artifacts that sit too close to a vertical-window cluster.
    verticals = [c for c in candidates if c.orientation == "vertical" and c.subtype == "window"]
    filtered: list[WindowDetection] = []
    for c in candidates:
        if c.subtype == "fix_window":
            close_vertical = any(
                (c.bbox[0] - 5.0 <= v.center[0] <= c.bbox[2] + 5.0)
                and abs(v.center[1] - c.center[1]) <= 40.0
                for v in verticals
            )
            if close_vertical:
                continue
        filtered.append(c)
    candidates = filtered

    candidates.sort(key=lambda x: -x.score)
    out: list[WindowDetection] = []
    for c in candidates:
        if any(_bbox_iou(c.bbox, e.bbox) > 0.5 for e in out):
            continue
        out.append(c)
    return out


def draw_windows(page: fitz.Page, windows: list[WindowDetection], out_path: str) -> None:
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image, ImageDraw
    import io

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    scale = 3.0
    for i, w in enumerate(windows):
        x0, y0, x1, y1 = [v * scale for v in w.bbox]
        color = (30, 144, 255) if w.subtype == "window" else (160, 60, 255)
        tag = "W" if w.subtype == "window" else "FW"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 14)), f"{tag}{i+1}", fill=color)
    img.save(out_path)
    print(f"  🖼️  Saved: {out_path} ({len(windows)} windows)")


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf/TLC_BZ商品計画テキスト2013モデルプラン.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc = fitz.open(pdf_path)
    pages = [page_arg] if page_arg is not None else range(len(doc))

    out_dir = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem

    all_results: dict[int, list[dict]] = {}
    for pi in pages:
        page = doc[pi]
        print(f"\n📄 Page {pi} / {len(doc)-1} ({page.rect.width:.0f}x{page.rect.height:.0f} pt)")
        wins = find_windows_topology(page, pi)
        n_fix = sum(1 for w in wins if w.subtype == "fix_window")
        print(f"  → {len(wins)} windows ({n_fix} fix)")
        if wins:
            out_png = out_dir / f"{stem}_page{pi}_windows_vector_topology.png"
            draw_windows(page, wins, str(out_png))
            all_results[pi] = [w.to_dict() for w in wins]

    out_json = out_dir / f"{stem}_windows_vector_topology.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
