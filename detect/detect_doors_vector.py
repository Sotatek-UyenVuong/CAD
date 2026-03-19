"""
Vector-based door symbol detector for CAD PDF drawings.

Phát hiện ký hiệu cửa đơn (片開き扉) trực tiếp từ vector path trong PDF —
không cần LLM, không cần render ảnh.

Nguyên lý:
  Cửa đơn = 1 đoạn thẳng (cánh cửa) + 1 cung 90° bezier (quỹ đạo mở)
  Hai phần chia sẻ cùng 1 điểm (điểm bản lề).
  Bán kính cung ≈ chiều dài cánh cửa.

Cách chạy:
  python detect_doors_vector.py <file.pdf> [page_index]  (default: tất cả trang)
"""

import sys
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import fitz  # pymupdf


# ── Tolerance settings ──────────────────────────────────────────────────────
ENDPOINT_TOL   = 8.0    # pt — khoảng cách tối đa để 2 điểm coi là "cùng điểm"
RADIUS_TOL     = 0.25   # tỉ lệ — |r_arc - len_line| / len_line ≤ tol
SWEEP_TOL_DEG  = 30.0   # độ — sai lệch cho phép với 90°
MIN_DOOR_PT    = 14.0   # pt — chiều dài cánh cửa tối thiểu (lọc toilet/fixture arcs)
MAX_DOOR_PT    = 30.0   # pt — chiều dài cánh cửa tối đa (lọc stair arcs, wall curves)
MIN_SWEEP_DEG  = 65.0   # độ — sweep tối thiểu (loại partial arcs)
MAX_SWEEP_DEG  = 110.0  # độ — sweep tối đa (loại arcs quá rộng)

# Hệ số bezier cho cung tròn 1/4 (κ = 4*(sqrt(2)-1)/3 ≈ 0.5523)
KAPPA = 4 * (math.sqrt(2) - 1) / 3


# ── Geometric helpers ────────────────────────────────────────────────────────
def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def dot(u, v) -> float:
    return u[0]*v[0] + u[1]*v[1]


def cross(u, v) -> float:
    return u[0]*v[1] - u[1]*v[0]


def norm(v) -> float:
    return math.hypot(v[0], v[1])


def near(a, b, tol=ENDPOINT_TOL) -> bool:
    return dist(a, b) <= tol


@dataclass
class LineSeg:
    p0: tuple
    p1: tuple

    @property
    def length(self) -> float:
        return dist(self.p0, self.p1)

    @property
    def endpoints(self):
        return (self.p0, self.p1)


@dataclass
class PolyArc:
    """
    Arc xấp xỉ bằng polyline (nhiều đoạn thẳng liên tiếp).
    PDF từ CAD thường dùng cách này thay vì bezier.
    """
    points: list   # danh sách điểm theo thứ tự
    center: tuple
    radius: float
    sweep:  float  # degrees, xấp xỉ 90

    @property
    def p0(self): return self.points[0]

    @property
    def p3(self): return self.points[-1]

    @property
    def endpoints(self):
        return (self.p0, self.p3)


def fit_circle(pts) -> Optional[tuple]:
    """
    Ước lượng tâm và bán kính từ tập điểm bằng least-squares.
    Trả về (cx, cy, r) hoặc None.
    """
    n = len(pts)
    if n < 3:
        return None
    samples = [pts[0], pts[n//2], pts[-1]]
    ax, ay = samples[0]
    bx, by = samples[1]
    cx, cy = samples[2]
    D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-9:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    r  = dist((ux, uy), samples[0])
    return ux, uy, r


# ── PDF path extractor ───────────────────────────────────────────────────────
def _drawing_points(drawing) -> list:
    """Lấy tất cả điểm theo thứ tự từ một drawing block."""
    pts = []
    for item in drawing.get("items", []):
        if item[0] == "l":
            p0 = (item[1].x, item[1].y)
            p1 = (item[2].x, item[2].y)
            if not pts or dist(pts[-1], p0) > 0.01:
                pts.append(p0)
            pts.append(p1)
    return pts


def _fit_circle_3pts(a, b, c) -> Optional[tuple]:
    ax, ay = a; bx, by = b; cx, cy = c
    D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-9:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    r  = math.hypot(ax-ux, ay-uy)
    return ux, uy, r


def _try_arc_from_tail(pts: list, max_skip: int = 3) -> Optional[PolyArc]:
    """
    Thử fit arc trên toàn path (skip=0) hoặc bỏ qua 1–max_skip điểm đầu (door leaf).
    skip=0: path là arc thuần (không có door leaf prefix).
    skip≥1: bỏ qua door leaf ở đầu path.
    tail=1: cũng thử bỏ qua điểm cuối (closing line về hinge).
    """
    for skip in range(0, min(max_skip + 1, len(pts) - 2)):
        for tail in (0, 1):
            end = len(pts) - tail if tail > 0 else None
            arc_pts = pts[skip:end]
            if len(arc_pts) < 4:
                continue
            mid = len(arc_pts) // 2
            result = _fit_circle_3pts(arc_pts[0], arc_pts[mid], arc_pts[-1])
            if result is None:
                continue
            cx, cy, radius = result
            if radius < MIN_DOOR_PT or radius > MAX_DOOR_PT:
                continue
            dists = [dist((cx, cy), p) for p in arc_pts]
            avg_d = sum(dists) / len(dists)
            scatter = max(abs(d - avg_d) for d in dists) / max(avg_d, 1e-6)
            if scatter > 0.20:
                continue
            a0 = math.degrees(math.atan2(arc_pts[0][1] - cy,  arc_pts[0][0] - cx))
            a1 = math.degrees(math.atan2(arc_pts[-1][1] - cy, arc_pts[-1][0] - cx))
            sweep = abs((a1 - a0 + 180) % 360 - 180)
            if not (MIN_SWEEP_DEG <= sweep <= MAX_SWEEP_DEG):
                continue
            return PolyArc(points=arc_pts, center=(cx, cy), radius=radius, sweep=sweep)
    return None


def extract_wall_lines(page: fitz.Page) -> list[LineSeg]:
    """Extract tất cả lines dài (tường) — không giới hạn chiều dài."""
    walls = []
    for drawing in page.get_drawings():
        r = drawing['rect']
        if r.width > 1 and r.height > 1:
            continue
        pts = _drawing_points(drawing)
        if len(pts) < 2:
            continue
        seg = LineSeg(pts[0], pts[-1])
        if seg.length > MIN_DOOR_PT * 2:
            walls.append(seg)
    return walls


def extract_paths(page: fitz.Page) -> tuple[list[LineSeg], list[PolyArc]]:
    lines: list[LineSeg] = []
    arcs:  list[PolyArc] = []

    for drawing in page.get_drawings():
        r = drawing['rect']
        w, h = r.width, r.height
        items = drawing.get("items", [])
        n_items = len(items)
        pts = _drawing_points(drawing)
        if len(pts) < 2:
            continue

        # ── Straight line: rất hẹp theo 1 chiều ──────────────────────
        if (w < ENDPOINT_TOL or h < ENDPOINT_TOL) and n_items <= 2:
            seg = LineSeg(pts[0], pts[-1])
            if MIN_DOOR_PT <= seg.length <= MAX_DOOR_PT:
                lines.append(seg)
            continue

        if n_items < 4:
            continue

        # ── Thử tìm arc trong phần tail của path (skip door leaf đầu) ──
        arc = _try_arc_from_tail(pts, max_skip=3)
        if arc is not None:
            arcs.append(arc)

    return lines, arcs


# ── Door symbol matcher ───────────────────────────────────────────────────────
@dataclass
class DoorSymbol:
    line:   LineSeg
    arc:    PolyArc
    hinge:  tuple   # điểm bản lề (chia sẻ giữa line và arc)
    radius: float
    bbox:   tuple   # (x0, y0, x1, y1) in PDF points

    def to_dict(self):
        return {
            "type":   "single",
            "hinge":  list(self.hinge),
            "radius": round(self.radius, 2),
            "bbox":   [round(v, 2) for v in self.bbox],
        }


@dataclass
class DoubleDoorSymbol:
    """両開き戸 — hai cánh cửa đối xứng, mỗi cánh xoay ra một phía."""
    door1: "DoorSymbol"
    door2: "DoorSymbol"
    bbox:  tuple   # union bbox

    def to_dict(self):
        return {
            "type":   "double",
            "hinge1": list(self.door1.hinge),
            "hinge2": list(self.door2.hinge),
            "radius": round((self.door1.radius + self.door2.radius) / 2, 2),
            "bbox":   [round(v, 2) for v in self.bbox],
        }


_wall_lines_cache: list[LineSeg] = []


def _bboxes_overlap(a: tuple, b: tuple, thr: float = 0.5) -> bool:
    """True nếu IoU của 2 bbox vượt thr."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter == 0:
        return False
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return inter / min(area_a, area_b) >= thr


def find_doors(lines: list[LineSeg], arcs: list[PolyArc]) -> list[DoorSymbol]:
    doors: list[DoorSymbol] = []

    for arc in arcs:
        cx, cy = arc.center
        r = arc.radius

        # Tìm door leaf: ưu tiên line riêng, fallback dùng synthetic từ center
        best_line: Optional[LineSeg] = None
        has_real_line = False
        best_hinge: tuple = arc.p0

        for line in lines:
            L = line.length
            if abs(L - r) / max(r, 1e-6) > RADIUS_TOL:
                continue
            for line_ep, other_ep in [(line.p0, line.p1), (line.p1, line.p0)]:
                for arc_ep in arc.endpoints:
                    if near(line_ep, arc_ep, ENDPOINT_TOL):
                        if near(other_ep, (cx, cy), ENDPOINT_TOL * 2):
                            best_line = line
                            best_hinge = other_ep
                            has_real_line = True
                            break
                if has_real_line:
                    break

        # Nếu không tìm được line riêng, dùng synthetic (center → arc endpoint)
        synthetic = best_line is None
        if synthetic:
            best_hinge = (cx, cy)
            best_line = LineSeg(best_hinge, arc.p0)

        # Kiểm tra hinge phải nằm gần arc center (bản lề = tâm cung)
        if dist(best_hinge, (cx, cy)) > r * 0.6:
            continue

        # Bounding box bao cả line + arc
        all_pts = [best_line.p0, best_line.p1] + arc.points
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # Synthetic door filters:
        #  • r ≥ 16.5pt (nhỏ hơn = closet symbol / fixture arc)
        #  • sweep 78°–100° (ngoài range = fixture/bathtub/bad fit, cửa thật luôn ≤ 95°)
        #  • min(bw,bh) ≥ r×0.78 (loại arc quá dẹt)
        #  • max(bw,bh) ≤ r×1.7 (loại arc fit sai — composite path / r bị phóng đại)
        if synthetic:
            if r < 16.5:
                continue
            if not (78.0 <= arc.sweep <= 100.0):
                continue
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if min(bw, bh) < r * 0.78:
                continue
            if max(bw, bh) > r * 1.7:
                continue

        doors.append(DoorSymbol(
            line=best_line,
            arc=arc,
            hinge=best_hinge,
            radius=r,
            bbox=bbox,
        ))

    # Dedup: loại trùng lặp theo bbox overlap > 50%
    deduped: list[DoorSymbol] = []
    for d in doors:
        if not any(_bboxes_overlap(d.bbox, e.bbox, thr=0.5) for e in deduped):
            deduped.append(d)

    return deduped


# ── 両開き戸 (double swing door) detector ────────────────────────────────────
def _leaf_tip(door: DoorSymbol) -> tuple:
    """Endpoint của cánh cửa (đầu xa bản lề)."""
    p0, p1 = door.line.p0, door.line.p1
    return p1 if dist(door.hinge, p0) < dist(door.hinge, p1) else p0


def _angle_pt(p1: tuple, p2: tuple) -> float:
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def _adiff(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def find_double_doors(
    single_doors: list[DoorSymbol],
) -> tuple[list[DoorSymbol], list[DoubleDoorSymbol]]:
    """
    Tìm cặp 両開き戸 từ danh sách 片開き戸.

    Điều kiện (đủ cả 5):
      • r1 ≈ r2 (trong vòng 15 %)
      • dist(hinge1, hinge2) ≈ r1 + r2 (trong vòng 10 %)
      • Cánh 1 chỉ hướng về phía hinge2 (lệch ≤ 25°)
      • Cánh 2 chỉ hướng về phía hinge1 (lệch ≤ 25°)
      • Đầu 2 cánh gần nhau: dist(tip1, tip2) ≤ 30 % × (r1+r2)
        → đây là điều kiện quyết định: 2 cánh CỬA GẶP nhau ở trung tâm

    Trả về: (remaining singles, double doors)
    """
    ANGLE_TOL  = math.radians(25)  # chặt hơn 40° cũ
    DIST_TOL   = 0.10              # ±10 % thay vì 12 %
    TIP_TOL    = 0.30              # đầu cánh cách nhau ≤ 30 % of r1+r2

    used = [False] * len(single_doors)
    doubles: list[DoubleDoorSymbol] = []

    for i, d1 in enumerate(single_doors):
        if used[i]:
            continue
        h1, r1 = d1.hinge, d1.radius
        t1 = _leaf_tip(d1)

        best_j, best_score = None, float('inf')

        for j, d2 in enumerate(single_doors):
            if j == i or used[j]:
                continue
            h2, r2 = d2.hinge, d2.radius

            # 1. Bán kính tương đương
            if abs(r1 - r2) > max(r1, r2) * 0.15:
                continue

            # 2. Khoảng cách bản lề ≈ r1 + r2
            d_h = dist(h1, h2)
            exp = r1 + r2
            if not ((1 - DIST_TOL) * exp <= d_h <= (1 + DIST_TOL) * exp):
                continue

            # 3. Cánh 1 hướng về hinge2
            ang_12 = _angle_pt(h1, h2)
            ang_l1 = _angle_pt(h1, t1)
            if _adiff(ang_l1, ang_12) > ANGLE_TOL:
                continue

            # 4. Cánh 2 hướng về hinge1
            t2 = _leaf_tip(d2)
            ang_21 = _angle_pt(h2, h1)
            ang_l2 = _angle_pt(h2, t2)
            if _adiff(ang_l2, ang_21) > ANGLE_TOL:
                continue

            # 5. Đầu 2 cánh phải GẶP NHAU (gặp ở trung tâm khung cửa)
            if dist(t1, t2) > TIP_TOL * exp:
                continue

            # 6. 2 bản lề phải nằm cùng TƯỜNG (offset vuông góc < 10 % của span)
            #    Nếu lệch > 10 %, là 2 cửa đơn riêng biệt ở 2 tường khác nhau.
            leaf_dx = t1[0] - h1[0]
            leaf_dy = t1[1] - h1[1]
            leaf_len = math.hypot(leaf_dx, leaf_dy)
            if leaf_len > 1e-6:
                ux, uy = leaf_dx / leaf_len, leaf_dy / leaf_len
                hdx, hdy = h2[0] - h1[0], h2[1] - h1[1]
                # Thành phần vuông góc với phương cánh cửa
                perp = abs(hdx * (-uy) + hdy * ux)
                if perp > 0.10 * exp:
                    continue

            score = _adiff(ang_l1, ang_12) + _adiff(ang_l2, ang_21)
            if score < best_score:
                best_score, best_j = score, j

        if best_j is not None:
            d2 = single_doors[best_j]
            used[i] = used[best_j] = True
            combined = (
                min(d1.bbox[0], d2.bbox[0]), min(d1.bbox[1], d2.bbox[1]),
                max(d1.bbox[2], d2.bbox[2]), max(d1.bbox[3], d2.bbox[3]),
            )
            doubles.append(DoubleDoorSymbol(door1=d1, door2=d2, bbox=combined))

    remaining = [d for i, d in enumerate(single_doors) if not used[i]]
    return remaining, doubles


# ── Draw results ─────────────────────────────────────────────────────────────
def draw_doors(page: fitz.Page, doors: list[DoorSymbol], out_path: str,
               double_doors: list[DoubleDoorSymbol] | None = None):
    """Render trang thành PNG và vẽ bbox lên."""
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)

    from PIL import Image, ImageDraw
    import io

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    scale = 3.0

    # 片開き戸 — red
    for i, d in enumerate(doors):
        x0, y0, x1, y1 = [v * scale for v in d.bbox]
        draw.rectangle([x0, y0, x1, y1], outline=(220, 20, 60), width=3)
        draw.text((x0, max(0, y0 - 14)), f"door {i+1}", fill=(220, 20, 60))

    # 両開き戸 — orange
    if double_doors:
        for i, dd in enumerate(double_doors):
            x0, y0, x1, y1 = [v * scale for v in dd.bbox]
            draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=4)
            draw.text((x0, max(0, y0 - 14)), f"double {i+1}", fill=(255, 140, 0))

    n_double = len(double_doors) if double_doors else 0
    img.save(out_path)
    print(f"  🖼️  Saved: {out_path}  ({len(doors)} single + {n_double} double)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf/TLC_BZ商品計画テキスト2013モデルプラン.pdf"
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    doc = fitz.open(pdf_path)
    total = len(doc)
    pages = [page_arg] if page_arg is not None else range(total)

    out_dir = Path("chatbot_exports")
    out_dir.mkdir(exist_ok=True)
    pdf_stem = Path(pdf_path).stem

    all_results = {}

    for pi in pages:
        page = doc[pi]
        print(f"\n📄 Page {pi} / {total-1}  ({page.rect.width:.0f}×{page.rect.height:.0f} pt)")

        lines, arcs = extract_paths(page)
        import detect_doors_vector as _self
        _self._wall_lines_cache = extract_wall_lines(page)
        print(f"  → {len(lines)} door-sized lines,  {len(arcs)} arcs,  {len(_self._wall_lines_cache)} wall lines")

        doors = find_doors(lines, arcs)
        singles, doubles = find_double_doors(doors)
        print(f"  → {len(singles)} single + {len(doubles)} double door symbols")

        if singles or doubles:
            out_png = out_dir / f"{pdf_stem}_page{pi}_doors_vector.png"
            draw_doors(page, singles, str(out_png), double_doors=doubles)
            all_results[pi] = (
                [d.to_dict() for d in singles] +
                [dd.to_dict() for dd in doubles]
            )

    out_json = out_dir / f"{pdf_stem}_doors_vector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {out_json}")


if __name__ == "__main__":
    main()
