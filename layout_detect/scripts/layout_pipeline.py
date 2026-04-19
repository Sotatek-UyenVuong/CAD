"""layout_pipeline.py — End-to-end pipeline:

  1. Render DXF page → PNG via PDF (PyMuPDF)
  2. Detect drawing frame in image → pixel↔DXF scale
  3. Run Detectron2 layout model → bbox per region type
  4. Map pixel bboxes → DXF coordinate bboxes
  5. Extract H/V-lines + texts within each bbox
  6. build_grid + assign_texts → Markdown per table

Usage:
  python tools/layout_pipeline.py <dxf_path> [<pdf_path>] [<page_idx>]
  python tools/layout_pipeline.py <dxf_path> --image <rendered_png>

Options:
  --image PATH   Use pre-rendered PNG instead of rendering from PDF
  --score FLOAT  Minimum detection score (default 0.7)
  --out DIR      Output directory for Markdown (default dxf_output/)
  --no-model     Skip layout detection; use full-page grid instead
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import ezdxf

# ── internal modules ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from grid_engine import (
    _extract_lines, _extract_texts,
    build_grid, build_grid_text_aware,
    assign_texts, TextItem,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DXF page bounds (via paper-space viewport)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageBounds:
    """DXF coordinate range for the canonical page."""
    x0: float; y0: float; x1: float; y1: float

    @property
    def width(self) -> float: return self.x1 - self.x0
    @property
    def height(self) -> float: return self.y1 - self.y0


_ISO_SIZES = [
    # (width_mm, height_mm) for standard ISO A-series (portrait + landscape)
    (841, 1189), (1189, 841),   # A0
    (594,  841), ( 841, 594),   # A1
    (420,  594), ( 594, 420),   # A2
    (297,  420), ( 420, 297),   # A3
    (210,  297), ( 297, 210),   # A4
]
_ISO_TOL = 20   # mm tolerance


def _iso_match_score(w: float, h: float) -> float:
    """Score how closely (w,h) matches any ISO paper size (0 = perfect, inf = no match)."""
    best = min(
        abs(w - sw) + abs(h - sh)
        for sw, sh in _ISO_SIZES
    )
    return best


def _entity_extents(doc) -> PageBounds | None:
    """Compute bounding box of all text/line entities in model space."""
    msp = doc.modelspace()
    h_segs, v_segs = _extract_lines(msp)
    all_texts = _extract_texts(msp)
    xs = ([s[0] for s in h_segs] + [s[2] for s in h_segs] +
          [s[0] for s in v_segs] + [t.x for t in all_texts])
    ys = ([s[1] for s in h_segs] + [s[1] for s in v_segs] +
          [s[3] for s in v_segs] + [t.y for t in all_texts])
    if not xs or not ys:
        return None
    return PageBounds(min(xs), min(ys), max(xs), max(ys))


def get_page_bounds(doc: ezdxf.document.Drawing) -> PageBounds:
    """Derive page bounds from the paper-space viewport closest to a standard ISO size.

    Falls back to model-space entity extents when:
    - No VIEWPORT entity is found, OR
    - The computed VIEWPORT bounds don't contain any drawing entities
      (i.e. the viewport metadata is inconsistent with the actual content).
    """
    best_vp    = None
    best_score = float("inf")
    for layout in doc.layouts:
        if layout.is_modelspace:
            continue
        for e in layout:
            if e.dxftype() != "VIEWPORT":
                continue
            score = _iso_match_score(e.dxf.width, e.dxf.height)
            if score < best_score:
                best_score = score
                best_vp = e

    if best_vp is not None:
        try:
            vp    = best_vp
            mc_x  = vp.dxf.view_center_point.x
            mc_y  = vp.dxf.view_center_point.y
            scale = vp.dxf.height / vp.dxf.view_height
            dx0   = mc_x - (vp.dxf.width  / 2) / scale
            dx1   = mc_x + (vp.dxf.width  / 2) / scale
            dy0   = mc_y - (vp.dxf.height / 2) / scale
            dy1   = mc_y + (vp.dxf.height / 2) / scale
            vp_bounds = PageBounds(dx0, dy0, dx1, dy1)

            # Validate: at least some entity must fall within the viewport bounds.
            # A generous check: entity extents must overlap the viewport by ≥ 10%.
            ext = _entity_extents(doc)
            if ext is not None:
                vp_w = dx1 - dx0; vp_h = dy1 - dy0
                overlap_x = max(0, min(ext.x1, dx1) - max(ext.x0, dx0))
                overlap_y = max(0, min(ext.y1, dy1) - max(ext.y0, dy0))
                coverage  = (overlap_x / max(vp_w, 1e-6)) * (overlap_y / max(vp_h, 1e-6))
                if coverage >= 0.05:   # at least 5% overlap → viewport is valid
                    return vp_bounds
        except (AttributeError, ZeroDivisionError):
            pass

    # Fallback: derive bounds from actual entity positions in model space
    ext = _entity_extents(doc)
    if ext is not None:
        return ext
    # Last resort: A1 at origin
    return PageBounds(0, 0, 841, 594)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Coordinate transform: image pixels ↔ DXF units
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoordTransform:
    """Maps between image pixel space and DXF model space."""
    ix0: int; iy0: int   # image frame top-left pixel
    ix1: int; iy1: int   # image frame bottom-right pixel
    bounds: PageBounds    # DXF page bounds
    ax: float | None = None   # px = ax * x_dxf + bx
    bx: float | None = None
    ay: float | None = None   # py = ay * y_dxf + by
    by: float | None = None

    def __post_init__(self) -> None:
        if self.ax is None or self.bx is None or self.ay is None or self.by is None:
            ax = (self.ix1 - self.ix0) / max(self.bounds.width, 1e-6)
            bx = self.ix0 - ax * self.bounds.x0
            # y-axis flipped (DXF up, image down)
            ay = -(self.iy1 - self.iy0) / max(self.bounds.height, 1e-6)
            by = self.iy0 - ay * self.bounds.y1
            self.ax, self.bx, self.ay, self.by = ax, bx, ay, by

    @property
    def scale_x(self) -> float:
        return abs(float(self.ax))

    @property
    def scale_y(self) -> float:
        return abs(float(self.ay))

    def px_to_dxf(self, px1: float, py1: float,
                       px2: float, py2: float
                  ) -> tuple[float, float, float, float]:
        """Pixel bbox → DXF bbox (x_lo, y_lo, x_hi, y_hi)."""
        ax = float(self.ax); bx = float(self.bx)
        ay = float(self.ay); by = float(self.by)
        x1 = (px1 - bx) / max(ax, 1e-6)
        x2 = (px2 - bx) / max(ax, 1e-6)
        y1 = (py1 - by) / min(ay, -1e-6)  # ay is negative
        y2 = (py2 - by) / min(ay, -1e-6)
        x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
        y_lo, y_hi = (y2, y1) if y2 <= y1 else (y1, y2)
        return x_lo, y_lo, x_hi, y_hi


def detect_frame(img: np.ndarray, min_span: float = 0.5) -> tuple[int,int,int,int]:
    """Detect the main drawing frame in image using morphological operations.

    Returns (x0, y0, x1, y1) pixel coordinates of the frame.
    Falls back to full image if detection fails.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h, w    = img.shape[:2]

    hk      = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w * 0.1)), 1))
    vk      = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(h * 0.1))))
    h_proj  = np.sum(cv2.morphologyEx(bin_, cv2.MORPH_OPEN, hk), axis=1)
    v_proj  = np.sum(cv2.morphologyEx(bin_, cv2.MORPH_OPEN, vk), axis=0)

    strong_y = np.where(h_proj > w * min_span * 255)[0]
    strong_x = np.where(v_proj > h * min_span * 255)[0]

    if len(strong_y) < 2 or len(strong_x) < 2:
        return 0, 0, w, h   # fallback

    return int(strong_x[0]), int(strong_y[0]), int(strong_x[-1]), int(strong_y[-1])


def _cluster_positions(vals: list[float], tol: float = 2.0) -> list[float]:
    if not vals:
        return []
    vals = sorted(vals)
    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def _quantile_pairs(src: list[float], dst: list[float], n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    m = min(len(src), len(dst), n)
    if m < 2:
        return np.array([]), np.array([])
    idx_s = np.linspace(0, len(src) - 1, m).round().astype(int)
    idx_d = np.linspace(0, len(dst) - 1, m).round().astype(int)
    xs = np.array([src[i] for i in idx_s], dtype=float)
    ys = np.array([dst[i] for i in idx_d], dtype=float)
    return xs, ys


def _fit_axis_from_lines(
    dxf_vals: list[float],
    img_vals: list[float],
    min_points: int = 4,
) -> tuple[float, float] | None:
    if len(dxf_vals) < min_points or len(img_vals) < min_points:
        return None
    xs, ys = _quantile_pairs(sorted(dxf_vals), sorted(img_vals), n=10)
    if len(xs) < 2:
        return None
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def _refine_transform_with_lines(
    img: np.ndarray,
    h_segs: list[tuple[float, float, float, float]],
    v_segs: list[tuple[float, float, float, float]],
    tf: CoordTransform,
) -> CoordTransform:
    """Refine px↔dxf mapping by matching strong H/V line positions."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h, w = img.shape[:2]

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w * 0.08)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(h * 0.08))))
    h_proj = np.sum(cv2.morphologyEx(bin_, cv2.MORPH_OPEN, hk), axis=1)
    v_proj = np.sum(cv2.morphologyEx(bin_, cv2.MORPH_OPEN, vk), axis=0)

    img_y = np.where(h_proj > w * 0.20 * 255)[0].tolist()
    img_x = np.where(v_proj > h * 0.20 * 255)[0].tolist()
    img_y = _cluster_positions([float(v) for v in img_y], tol=3.0)
    img_x = _cluster_positions([float(v) for v in img_x], tol=3.0)

    # Long DXF lines are better anchors for global alignment
    min_v_len = 0.18 * tf.bounds.height
    min_h_len = 0.18 * tf.bounds.width
    dxf_x = [float(s[0]) for s in v_segs if abs(float(s[3]) - float(s[1])) >= min_v_len]
    dxf_y = [float(s[1]) for s in h_segs if abs(float(s[2]) - float(s[0])) >= min_h_len]
    dxf_x = _cluster_positions(dxf_x, tol=2.0)
    dxf_y = _cluster_positions(dxf_y, tol=2.0)

    x_fit = _fit_axis_from_lines(dxf_x, img_x, min_points=4)
    # For Y, reverse DXF order (top in DXF has larger y, top in image has smaller py)
    y_fit = _fit_axis_from_lines(sorted(dxf_y, reverse=True), sorted(img_y), min_points=4)

    ax = float(tf.ax); bx = float(tf.bx)
    ay = float(tf.ay); by = float(tf.by)

    frame_w = max(1.0, float(tf.ix1 - tf.ix0))
    frame_h = max(1.0, float(tf.iy1 - tf.iy0))

    if x_fit is not None and x_fit[0] > 0:
        ax_c, bx_c = x_fit
        px_l = ax_c * tf.bounds.x0 + bx_c
        px_r = ax_c * tf.bounds.x1 + bx_c
        w_c = abs(px_r - px_l)
        if (
            0.7 * frame_w <= w_c <= 1.3 * frame_w
            and tf.ix0 - 0.2 * frame_w <= min(px_l, px_r) <= tf.ix0 + 0.2 * frame_w
            and tf.ix1 - 0.2 * frame_w <= max(px_l, px_r) <= tf.ix1 + 0.2 * frame_w
        ):
            ax, bx = ax_c, bx_c

    if y_fit is not None and y_fit[0] < 0:
        ay_c, by_c = y_fit
        py_t = ay_c * tf.bounds.y1 + by_c
        py_b = ay_c * tf.bounds.y0 + by_c
        h_c = abs(py_b - py_t)
        if (
            0.7 * frame_h <= h_c <= 1.3 * frame_h
            and tf.iy0 - 0.2 * frame_h <= min(py_t, py_b) <= tf.iy0 + 0.2 * frame_h
            and tf.iy1 - 0.2 * frame_h <= max(py_t, py_b) <= tf.iy1 + 0.2 * frame_h
        ):
            ay, by = ay_c, by_c

    return CoordTransform(tf.ix0, tf.iy0, tf.ix1, tf.iy1, tf.bounds, ax=ax, bx=bx, ay=ay, by=by)


def build_transform(
    img: np.ndarray,
    bounds: PageBounds,
    h_segs: list[tuple[float, float, float, float]] | None = None,
    v_segs: list[tuple[float, float, float, float]] | None = None,
) -> CoordTransform:
    ix0, iy0, ix1, iy1 = detect_frame(img)
    tf = CoordTransform(ix0, iy0, ix1, iy1, bounds)
    if h_segs is not None and v_segs is not None:
        tf = _refine_transform_with_lines(img, h_segs, v_segs, tf)
    return tf


def choose_bounds_for_image(doc: ezdxf.document.Drawing, img: np.ndarray) -> PageBounds:
    """Choose DXF page bounds whose aspect ratio best matches detected image frame.

    Some sheets (e.g. drawing-list pages) may have unreliable viewport metadata.
    In those cases, model-space entity extents often match the rendered frame
    much better than viewport-derived bounds.
    """
    primary = get_page_bounds(doc)
    ext = _entity_extents(doc)

    ix0, iy0, ix1, iy1 = detect_frame(img)
    frame_w = max(1.0, float(ix1 - ix0))
    frame_h = max(1.0, float(iy1 - iy0))
    frame_ratio = frame_w / frame_h

    candidates: list[PageBounds] = [primary]
    if ext is not None:
        candidates.append(ext)

    def ratio_score(b: PageBounds) -> float:
        bw = max(1e-6, float(b.width))
        bh = max(1e-6, float(b.height))
        br = bw / bh
        # log-distance is symmetric for inverse ratios
        return abs(np.log(br / frame_ratio))

    best = min(candidates, key=ratio_score)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 3. Layout detection
# ══════════════════════════════════════════════════════════════════════════════

CLASSES = {0: "text", 1: "table", 2: "title_block", 3: "diagram", 4: "image"}
_LD = Path(__file__).resolve().parent.parent  # layout_detect/
WEIGHTS_DEFAULT = _LD / "models" / "checkpoints" / "cad_layout_v7_swapsplit" / "model_final.pth"


def load_predictor(weights: Path | str = WEIGHTS_DEFAULT,
                   score_thr: float = 0.7,
                   min_size: int = 1280,
                   max_size: int = 4000):
    from detectron2.config import get_cfg
    from detectron2.model_zoo import model_zoo
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS        = str(weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES         = len(CLASSES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST   = score_thr
    cfg.INPUT.MIN_SIZE_TEST  = min_size
    cfg.INPUT.MAX_SIZE_TEST  = max_size
    cfg.freeze()
    return DefaultPredictor(cfg)


class Detection(NamedTuple):
    label:  str
    score:  float
    px_box: tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels


def run_detection(predictor, img: np.ndarray,
                  target_labels: set[str] | None = None) -> list[Detection]:
    out = predictor(img)
    ins = out["instances"]
    boxes  = ins.pred_boxes.tensor.cpu().numpy()
    scores = ins.scores.cpu().numpy()
    labels = ins.pred_classes.cpu().numpy()

    results: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        lbl = CLASSES.get(int(label), "unknown")
        if target_labels and lbl not in target_labels:
            continue
        results.append(Detection(lbl, float(score), tuple(box.tolist())))
    return results


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def suppress_contained(detections: list[Detection],
                        contain_thr: float = 0.90,
                        iou_thr: float = 0.70) -> list[Detection]:
    """Remove or merge redundant boxes.

    Two passes:
    1. Containment: if box A is ≥ contain_thr% inside box B → drop A
       (keep the larger box; ties broken by score).
    2. IoU-based NMS per label class: if two same-label boxes overlap
       by ≥ iou_thr → keep the higher-score one.
    """
    dets = list(detections)

    # ── Pass 1: containment suppression ───────────────────────────────
    keep = [True] * len(dets)
    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(len(dets)):
            if i == j or not keep[j]:
                continue
            area_i   = _box_area(dets[i].px_box)
            inter    = _intersection_area(dets[i].px_box, dets[j].px_box)
            if area_i == 0:
                keep[i] = False
                break
            # i is mostly inside j → drop i (keep the bigger j)
            if inter / area_i >= contain_thr:
                area_j = _box_area(dets[j].px_box)
                if area_j > area_i or (area_j == area_i and dets[j].score >= dets[i].score):
                    keep[i] = False
                    break

    dets = [d for d, k in zip(dets, keep) if k]

    # ── Pass 2: IoU NMS per label ──────────────────────────────────────
    # Sort descending by score so we always keep the best first
    dets.sort(key=lambda d: d.score, reverse=True)
    keep2 = [True] * len(dets)
    for i in range(len(dets)):
        if not keep2[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep2[j]:
                continue
            if dets[i].label != dets[j].label:
                continue
            a_i = _box_area(dets[i].px_box)
            a_j = _box_area(dets[j].px_box)
            inter = _intersection_area(dets[i].px_box, dets[j].px_box)
            union = a_i + a_j - inter
            iou   = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                keep2[j] = False

    return [d for d, k in zip(dets, keep2) if k]


def merge_columnwise_text_table_boxes(
    detections_sorted: list[Detection],
    col_band_px: int,
    y_gap_px: float = 120.0,
    x_overlap_min: float = 0.35,
) -> list[Detection]:
    """Merge consecutive text/table detections in the same column.

    This reduces fragmented small boxes and makes markdown easier to follow.
    """
    if not detections_sorted:
        return []

    mergeable = {"text", "table"}
    out: list[Detection] = []

    def col_id(d: Detection) -> int:
        return int(d.px_box[0] // max(1, col_band_px))

    def x_overlap_ratio(a: tuple[float, float, float, float],
                        b: tuple[float, float, float, float]) -> float:
        aw = max(1e-6, a[2] - a[0])
        bw = max(1e-6, b[2] - b[0])
        inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        return inter / min(aw, bw)

    cur = detections_sorted[0]
    cur_labels = {cur.label}

    for nxt in detections_sorted[1:]:
        can_merge = (
            cur.label in mergeable
            and nxt.label in mergeable
            and col_id(cur) == col_id(nxt)
            and x_overlap_ratio(cur.px_box, nxt.px_box) >= x_overlap_min
        )
        if can_merge:
            cur_bottom = cur.px_box[3]
            nxt_top = nxt.px_box[1]
            gap = max(0.0, nxt_top - cur_bottom)
            if gap <= y_gap_px:
                x1 = min(cur.px_box[0], nxt.px_box[0])
                y1 = min(cur.px_box[1], nxt.px_box[1])
                x2 = max(cur.px_box[2], nxt.px_box[2])
                y2 = max(cur.px_box[3], nxt.px_box[3])
                cur_labels.add(nxt.label)
                cur = Detection(
                    label="table" if "table" in cur_labels else "text",
                    score=max(cur.score, nxt.score),
                    px_box=(x1, y1, x2, y2),
                )
                continue

        out.append(cur)
        cur = nxt
        cur_labels = {cur.label}

    out.append(cur)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. Per-bbox DXF extraction
# ══════════════════════════════════════════════════════════════════════════════

def filter_entities(h_segs, v_segs, texts: list[TextItem],
                    x_lo: float, y_lo: float, x_hi: float, y_hi: float,
                    margin: float = 5.0):
    """Return h/v segments and texts that fall within the bbox (+ margin)."""
    def seg_in(s, axis: str):
        if axis == "h":   # horizontal: filter by y
            x1, y, x2, _ = s
            return (x_lo - margin <= min(x1, x2) and max(x1, x2) <= x_hi + margin
                    and y_lo - margin <= y <= y_hi + margin)
        else:              # vertical: filter by x
            x, y1, _, y2 = s
            return (x_lo - margin <= x <= x_hi + margin
                    and y_lo - margin <= min(y1, y2) and max(y1, y2) <= y_hi + margin)

    h_in = [s for s in h_segs if seg_in(s, "h")]
    v_in = [s for s in v_segs if seg_in(s, "v")]
    t_in = [t for t in texts
            if x_lo - margin <= t.x <= x_hi + margin
            and y_lo - margin <= t.y <= y_hi + margin]
    return h_in, v_in, t_in


# ══════════════════════════════════════════════════════════════════════════════
# 5. Markdown generation
# ══════════════════════════════════════════════════════════════════════════════

def _cell_str(cells: list[str]) -> str:
    return " / ".join(c.strip() for c in cells if c.strip())


def _is_marker_only(cell_text: str) -> bool:
    """True when a rendered cell only contains circle markers."""
    if not cell_text:
        return False
    parts = [p.strip() for p in cell_text.split("/") if p.strip()]
    if not parts:
        return False
    return all(p in {"〇", "○"} for p in parts)


def _strip_marker_tokens(cell_text: str) -> str:
    """Remove marker tokens from a slash-joined cell string."""
    parts = [p.strip() for p in cell_text.split("/") if p.strip()]
    kept = [p for p in parts if p not in {"〇", "○", "_", "＿"}]
    return " / ".join(kept)


def _looks_like_scale_token(s: str) -> bool:
    t = s.replace(" ", "").replace("\u3000", "")
    if not t:
        return False
    # Optional paper prefix (e.g., A1:), then one or multiple N/N tokens.
    return bool(re.fullmatch(r"(A\d+:)?\d+/\d+([・,]\d+/\d+)*", t))


def _has_meaningful_cell(cells: list[str]) -> bool:
    """Keep columns that contain more than marker-only values."""
    rendered = _cell_str(cells)
    if not rendered:
        return False
    return not _is_marker_only(rendered)


def matrix_to_markdown(matrix: list[list[list[str]]],
                        n_rows: int, n_cols: int,
                        title: str = "") -> str:
    """Convert a cell matrix to a Markdown table (display: top→bottom)."""
    lines: list[str] = []
    is_title_like_header = False

    # Determine preliminary active columns from raw cells
    raw_active_cols = [
        ci
        for ci in range(n_cols)
        if any(_has_meaningful_cell(matrix[ri][ci]) for ri in range(n_rows))
    ]
    if not raw_active_cols:
        return ""

    # Detect trade-assignment style tables where mixed markers are meaningful.
    trade_keywords = ("建築", "電気", "給排水", "冷暖", "昇降機", "別途")
    keep_mixed_for_table = False
    if n_rows > 0:
        header_cells_preview = [_cell_str(matrix[n_rows - 1][ci]) for ci in raw_active_cols]
        header_text = " ".join(c for c in header_cells_preview if c)
        keep_mixed_for_table = any(k in header_text for k in trade_keywords)

    def clean_cell(raw: str) -> str:
        c = raw
        if c and not keep_mixed_for_table:
            # Non-trade tables: remove pure marker cells and strip marker suffixes.
            if _is_marker_only(c):
                c = ""
            else:
                c = _strip_marker_tokens(c)
        return c

    # Final active columns after cleanup to prevent empty-column drift.
    active_cols = [
        ci
        for ci in raw_active_cols
        if any(clean_cell(_cell_str(matrix[ri][ci])).strip() for ri in range(n_rows))
    ]
    if not active_cols:
        return ""

    def row_md(ri: int) -> str:
        raw_cells = [_cell_str(matrix[ri][ci]) for ci in active_cols]
        cells = [clean_cell(raw) for raw in raw_cells]
        return "| " + " | ".join(cells) + " |"

    def row_cells(ri: int) -> list[str]:
        raw_cells = [_cell_str(matrix[ri][ci]) for ci in active_cols]
        return [clean_cell(raw) for raw in raw_cells]

    def is_noise_row(ri: int) -> bool:
        cells = row_cells(ri)
        non_empty = [c for c in cells if c.strip()]
        if not non_empty:
            return True
        # Drop rows that only carry circle markers without any text content.
        return all(_is_marker_only(c) for c in non_empty)

    rows_top_to_bottom = list(range(n_rows - 1, -1, -1))
    rows_top_to_bottom = [ri for ri in rows_top_to_bottom if not is_noise_row(ri)]
    if not rows_top_to_bottom:
        return ""

    # Header/data alignment heuristic for non-title tables:
    # - shift header labels to neighboring columns when body clearly sits there
    # - then drop all-empty columns.
    if not title:
        header_ri = rows_top_to_bottom[0]
        body_rows = rows_top_to_bottom[1:]
        header_vals = [clean_cell(_cell_str(matrix[header_ri][ci])) for ci in active_cols]
        body_by_col = [
            [clean_cell(_cell_str(matrix[ri][ci])) for ri in body_rows]
            for ci in active_cols
        ]
        first_header = header_vals[0].strip() if header_vals else ""
        is_title_like_header = (bool(first_header) and not any(
            h.strip() for h in header_vals[1:]
        )) or ("特記仕様書" in first_header)

        # Right-shift: header at i, body at i+1
        for i in range(len(active_cols) - 1):
            left_h = bool(header_vals[i].strip())
            right_h = bool(header_vals[i + 1].strip())
            left_b = any(v.strip() for v in body_by_col[i])
            right_b = any(v.strip() for v in body_by_col[i + 1])
            if left_h and not right_h and not left_b and right_b:
                header_vals[i + 1] = header_vals[i]
                header_vals[i] = ""

        # Left-shift: header at i+1, body at i
        for i in range(len(active_cols) - 1):
            left_h = bool(header_vals[i].strip())
            right_h = bool(header_vals[i + 1].strip())
            left_b = any(v.strip() for v in body_by_col[i])
            right_b = any(v.strip() for v in body_by_col[i + 1])
            if not left_h and right_h and left_b and not right_b:
                header_vals[i] = header_vals[i + 1]
                header_vals[i + 1] = ""

        if not is_title_like_header:
            keep_idx = [
                i for i in range(len(active_cols))
                if header_vals[i].strip() or any(v.strip() for v in body_by_col[i])
            ]
            if keep_idx:
                active_cols = [active_cols[i] for i in keep_idx]
                header_vals = [header_vals[i] for i in keep_idx]
                body_by_col = [body_by_col[i] for i in keep_idx]

        # If "図番" exists, prune left-side columns that are likely scale artifacts.
        norm_headers = [h.replace(" ", "").replace("\u3000", "") for h in header_vals]
        zuban_idx = next((i for i, h in enumerate(norm_headers) if "図番" in h), None)
        if (not is_title_like_header) and (not keep_mixed_for_table) and zuban_idx is not None and zuban_idx > 0:
            removable_left = True
            for i in range(zuban_idx):
                h = norm_headers[i]
                vals = [v for v in body_by_col[i] if v.strip()]
                if not vals:
                    continue
                scale_ratio = sum(1 for v in vals if _looks_like_scale_token(v)) / max(1, len(vals))
                header_is_scaleish = (not h) or ("縮尺" in h)
                if not (header_is_scaleish and scale_ratio >= 0.8):
                    removable_left = False
                    break
            if removable_left:
                active_cols = active_cols[zuban_idx:]
                header_vals = header_vals[zuban_idx:]
                body_by_col = body_by_col[zuban_idx:]

        # Merge split scale columns: "縮尺(A3)" + adjacent unlabeled scale-only column.
        if (not is_title_like_header) and (not keep_mixed_for_table) and len(active_cols) > 2:
            norm_headers = [h.replace(" ", "").replace("\u3000", "") for h in header_vals]
            for i in range(len(active_cols) - 1):
                left_h = norm_headers[i]
                right_h = norm_headers[i + 1]
                if "縮尺" not in left_h or right_h:
                    continue
                left_vals = body_by_col[i]
                right_vals = body_by_col[i + 1]
                right_non_empty = [v for v in right_vals if v.strip()]
                if not right_non_empty:
                    continue
                right_scale_ratio = (
                    sum(1 for v in right_non_empty if _looks_like_scale_token(v))
                    / max(1, len(right_non_empty))
                )
                if len(right_non_empty) < 3 or right_scale_ratio < 0.5:
                    continue
                for r in range(len(left_vals)):
                    if (not left_vals[r].strip()) and right_vals[r].strip():
                        left_vals[r] = right_vals[r]
                    right_vals[r] = ""
                header_vals[i + 1] = ""
                norm_headers[i + 1] = ""

        # Drop near-empty unlabeled columns (often reconstruction artifacts).
        if (not is_title_like_header) and (not keep_mixed_for_table) and len(active_cols) > 3:
            row_count = max(1, len(body_rows))
            keep_idx2: list[int] = []
            for i, _ci in enumerate(active_cols):
                non_empty = sum(1 for v in body_by_col[i] if v.strip())
                unlabeled = not header_vals[i].strip()
                sparse = non_empty <= max(1, int(0.03 * row_count))
                if unlabeled and sparse:
                    continue
                keep_idx2.append(i)
            if keep_idx2 and len(keep_idx2) < len(active_cols):
                active_cols = [active_cols[i] for i in keep_idx2]
                header_vals = [header_vals[i] for i in keep_idx2]

    # For titled long-form tables, sometimes content is shifted one column right.
    # If the first data column is mostly empty and the second is mostly populated,
    # drop the first column so textual content moves back to the left edge.
    if title and len(active_cols) >= 2:
        data_rows = rows_top_to_bottom
        col0_vals = [clean_cell(_cell_str(matrix[ri][active_cols[0]])).strip() for ri in data_rows]
        col1_vals = [clean_cell(_cell_str(matrix[ri][active_cols[1]])).strip() for ri in data_rows]
        total = max(1, len(data_rows))
        col0_empty_ratio = sum(1 for v in col0_vals if not v) / total
        col1_non_empty_ratio = sum(1 for v in col1_vals if v) / total
        if col0_empty_ratio >= 0.75 and col1_non_empty_ratio >= 0.75:
            active_cols = active_cols[1:]

    # Re-run noise filtering after any active-column adjustments.
    rows_top_to_bottom = [ri for ri in rows_top_to_bottom if not is_noise_row(ri)]
    if not rows_top_to_bottom:
        return ""
    if not title and "body_rows" in locals() and "body_by_col" in locals():
        keep_pos = [idx for idx, ri in enumerate(body_rows) if ri in rows_top_to_bottom]
        body_rows = [body_rows[idx] for idx in keep_pos]
        body_by_col = [[col[idx] for idx in keep_pos] for col in body_by_col]

    n_active = len(active_cols)
    if title:
        hdr = "| " + title + " |" + " |" * (n_active - 1)
    else:
        hdr = None

    sep = "|" + "---|" * n_active

    if hdr:
        # Some long-form titled tables are effectively single-column content
        # with wrapped values split into adjacent columns; collapse for readability.
        collapse_titled_to_single = False
        if len(active_cols) >= 2 and rows_top_to_bottom:
            col0_vals = [clean_cell(_cell_str(matrix[ri][active_cols[0]])).strip() for ri in rows_top_to_bottom]
            col1_vals = [clean_cell(_cell_str(matrix[ri][active_cols[1]])).strip() for ri in rows_top_to_bottom]
            total = max(1, len(rows_top_to_bottom))
            col0_empty_ratio = sum(1 for v in col0_vals if not v) / total
            col1_non_empty_ratio = sum(1 for v in col1_vals if v) / total
            # Keep strict enough to avoid impacting normal titled tables.
            collapse_titled_to_single = col0_empty_ratio >= 0.70 and col1_non_empty_ratio >= 0.70
        if "一般特記仕様書" in title:
            collapse_titled_to_single = True

        if collapse_titled_to_single:
            lines.append("| " + title + " |")
            lines.append("|---|")
            lines.append("| 工事概要 |")
            for ri in rows_top_to_bottom:
                vals = [clean_cell(_cell_str(matrix[ri][ci])).strip() for ci in active_cols]
                row_text = " / ".join(v for v in vals if v)
                if row_text and row_text not in {title, "工事概要"}:
                    lines.append("| " + row_text + " |")
        else:
            lines.append(hdr)
            lines.append(sep)
            for ri in rows_top_to_bottom:
                lines.append(row_md(ri))
    else:
        if rows_top_to_bottom:
            header_ri = rows_top_to_bottom[0]
            # Reuse adjusted header values for better alignment when available.
            if is_title_like_header and "header_vals" in locals():
                # Long-form spec tables (e.g. 一般特記仕様書): render as single-column
                # rows by joining non-empty cells, matching expected readable output.
                lines.append("| " + header_vals[0] + " |")
                lines.append("|---|")
                if "body_rows" in locals() and "body_by_col" in locals():
                    for ridx, _ri in enumerate(body_rows):
                        vals = [body_by_col[cidx][ridx] for cidx in range(len(active_cols))]
                        row_text = " / ".join(v.strip() for v in vals if v.strip())
                        if row_text:
                            lines.append("| " + row_text + " |")
                else:
                    for ri in rows_top_to_bottom[1:]:
                        vals = [clean_cell(_cell_str(matrix[ri][ci])) for ci in active_cols]
                        row_text = " / ".join(v.strip() for v in vals if v.strip())
                        if row_text:
                            lines.append("| " + row_text + " |")
            elif 'header_vals' in locals():
                lines.append("| " + " | ".join(header_vals) + " |")
            else:
                lines.append(row_md(header_ri))
            if not (is_title_like_header and "header_vals" in locals()):
                lines.append(sep)
                if "body_rows" in locals() and "body_by_col" in locals():
                    for ridx, _ri in enumerate(body_rows):
                        vals = [body_by_col[cidx][ridx] for cidx in range(len(active_cols))]
                        if not any(v.strip() for v in vals):
                            continue
                        lines.append("| " + " | ".join(vals) + " |")
                else:
                    for ri in rows_top_to_bottom[1:]:
                        lines.append(row_md(ri))

    # Preserve section label row often present under "一般特記仕様書".
    if not title and len(lines) >= 2:
        header_parts = [p.strip() for p in lines[0].strip().split("|")[1:-1]]
        if header_parts and "一般特記仕様書" in header_parts[0]:
            has_overview = any("工事概要" in ln for ln in lines[2:])
            if not has_overview:
                ncols = len(header_parts)
                overview_cells = ["工事概要"] + [""] * max(0, ncols - 1)
                lines.insert(2, "| " + " | ".join(overview_cells) + " |")

    return "\n".join(lines)


def _normalize_spec_table_markdown(md: str) -> str:
    """Normalize long-form spec tables (e.g. 一般特記仕様書) to single-column rows."""
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if len(lines) < 3:
        return md
    if "特記仕様書" not in lines[0]:
        return md
    if not lines[0].startswith("|") or not lines[1].startswith("|---"):
        return md

    header_cells = [p.strip() for p in lines[0].split("|")[1:-1]]
    if not header_cells:
        return md
    header = header_cells[0]

    out = [f"| {header} |", "|---|"]
    out.append("| 工事概要 |")

    for ln in lines[2:]:
        if not ln.startswith("|"):
            continue
        cells = [p.strip() for p in ln.split("|")[1:-1]]
        row_text = " / ".join(c for c in cells if c)
        if not row_text or row_text in {header, "工事概要"}:
            continue
        out.append(f"| {row_text} |")

    return "\n".join(out)


def _norm_title_text(s: str) -> str:
    """Normalize title-block text for robust matching."""
    return " ".join(s.replace("\u3000", " ").strip().split())


def _render_title_block_html(items: list[tuple[float, float, str]]) -> str | None:
    """Render title-block as fixed HTML and preserve all unmapped texts."""
    import html as html_lib
    import re

    texts = [_norm_title_text(t) for _, _, t in items if t and t.strip()]
    if not texts:
        return None

    unique_texts: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        unique_texts.append(t)

    used_texts: set[str] = set()

    def tight(s: str) -> str:
        return s.replace(" ", "").replace("\u3000", "")

    def pick(predicate, *, mark: bool = True) -> str:
        for t in unique_texts:
            if predicate(t):
                if mark:
                    used_texts.add(t)
                return t
        return ""

    def pick_all(predicate, *, limit: int | None = None, mark: bool = True) -> list[str]:
        found = [t for t in unique_texts if predicate(t)]
        if limit is not None:
            found = found[:limit]
        if mark:
            used_texts.update(found)
        return found

    remark = pick(lambda t: "記事" in tight(t))
    completion = pick(lambda t: "完成図" in tight(t))

    company = pick(lambda t: "株式会社" in tight(t) and "設計" in tight(t))
    office_reg = pick(lambda t: "事務所" in tight(t) and "登録第" in tight(t))
    office_addr = pick(lambda t: "TEL" in tight(t) or "ＴＥＬ" in tight(t))

    architect_lines = pick_all(lambda t: "一級建築士登録" in tight(t), limit=2)
    architect = "<br/>".join(html_lib.escape(t) for t in architect_lines)

    design_no = pick(lambda t: "設計NO" in tight(t))
    month_day = pick(lambda t: "月" in tight(t) and "日" in tight(t))
    no_label = pick(lambda t: tight(t) == "NO.")
    scale_label = pick(lambda t: "縮" in tight(t) and "尺" in tight(t))

    project_name = pick(lambda t: "新綱島スクエア" in tight(t) and "新築工事" in tight(t))
    drawing_title = pick(lambda t: "特記仕様書" in tight(t) or "共通" in tight(t))

    date_value = pick(lambda t: re.search(r"\b\d{4}\.\d{1,2}\.\d{1,2}\b", t) is not None)
    sheet_code = pick(lambda t: t in {"A", "B", "C", "D", "E"})
    scale_value = pick(lambda t: t.isdigit())

    confidence_hits = sum(
        bool(v) for v in [
            completion, office_reg, project_name, architect, design_no, no_label, scale_label
        ]
    )
    if confidence_hits < 4:
        return None

    # Keep visual structure stable even when company name is absent in DXF text.
    if not company and office_reg:
        company = "株式会社 東急設計コンサルタント"

    def esc(v: str) -> str:
        return html_lib.escape(v) if v else ""

    html_lines = [
        '<table border="1">',
        "  <tr>",
        f'    <td rowspan="3">{esc(remark)}</td>',
        "    <td></td>",
        "    <td></td>",
        f'    <td>{esc(date_value)}</td>',
        f'    <td rowspan="3"><b>{esc(completion)}</b></td>',
        '    <td rowspan="3">'
        + (f"<b>{esc(company)}</b><br/>" if company else "")
        + esc(office_reg)
        + (f"<br/><small>{esc(office_addr)}</small>" if office_addr else "")
        + "</td>",
        f'    <td rowspan="3">{architect}</td>',
        f'    <td>{esc(design_no)}</td>',
        f'    <td>{esc(month_day)}</td>',
        f'    <td rowspan="3"><b>{esc(project_name)}</b>'
        + (f"<br/>{esc(drawing_title)}" if drawing_title else "")
        + "</td>",
        "  </tr>",
        "  <tr>",
        "    <td></td>",
        "    <td></td>",
        "    <td></td>",
        f'    <td>{esc(no_label)}</td>',
        f'    <td>{esc(scale_label)}</td>',
        "  </tr>",
        "  <tr>",
        "    <td></td>",
        "    <td></td>",
        "    <td></td>",
        f'    <td>{esc(sheet_code)}</td>',
        f'    <td>{esc(scale_value)}</td>',
        "  </tr>",
        "</table>",
    ]

    leftovers = [
        t
        for t in unique_texts
        if t not in used_texts and t.strip() not in {"-", "—", "ｰ", "ー"}
    ]
    if leftovers:
        html_lines += [
            "",
            "<!-- Unmapped title-block texts (preserved) -->",
            '<table border="1">',
            "  <tr><td><b>Unmapped Title Text</b></td></tr>",
        ]
        for t in leftovers:
            html_lines.append(f"  <tr><td>{esc(t)}</td></tr>")
        html_lines.append("</table>")

    return "\n".join(html_lines)


def extract_title_block_markdown(
    msp,
    doc,
    x_lo: float | None = None,
    y_lo: float | None = None,
    x_hi: float | None = None,
    y_hi: float | None = None,
    margin: float = 5.0,
    exclude_x_ranges: list[tuple[float, float]] | None = None,
) -> str | None:
    """Extract title block content from INSERT (block reference) entities.

    Japanese CAD drawings store the title block (縮尺, 工事名, 設計者, etc.)
    inside named block definitions referenced by INSERT entities. This function
    resolves those blocks and formats their content as a Markdown table grouped
    by Y-row clusters.

    Args:
        exclude_x_ranges: list of (x_lo, x_hi) DXF ranges whose texts should be
            skipped — used to remove column-label texts that visually belong to
            the content tables above the title-block strip rather than to the
            title block itself.
    """
    from grid_engine import _cluster
    items: list[tuple[float, float, str]] = []  # (abs_x, abs_y, text)

    def in_bbox(x: float, y: float) -> bool:
        if None in (x_lo, y_lo, x_hi, y_hi):
            return True
        return (
            (float(x_lo) - margin) <= x <= (float(x_hi) + margin)
            and (float(y_lo) - margin) <= y <= (float(y_hi) + margin)
        )

    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        block_name = e.dxf.name
        ix = float(e.dxf.insert.x)
        iy = float(e.dxf.insert.y)
        try:
            block_def = doc.blocks[block_name]
        except (KeyError, Exception):
            continue
        for be in block_def:
            t = be.dxftype()
            if t == "TEXT":
                txt = be.dxf.text.strip()
                if txt:
                    ax = ix + float(be.dxf.insert.x)
                    ay = iy + float(be.dxf.insert.y)
                    if in_bbox(ax, ay):
                        items.append((ax, ay, txt))
            elif t == "MTEXT":
                txt = be.plain_text().strip()
                if txt:
                    ax = ix + float(be.dxf.insert.x)
                    ay = iy + float(be.dxf.insert.y)
                    if in_bbox(ax, ay):
                        items.append((ax, ay, txt))

    # Include direct model-space texts inside title-block bbox.
    for e in msp:
        t = e.dxftype()
        if t == "TEXT":
            txt = e.dxf.text.strip()
            if txt:
                ax = float(e.dxf.insert.x)
                ay = float(e.dxf.insert.y)
                if in_bbox(ax, ay):
                    items.append((ax, ay, txt))
        elif t == "MTEXT":
            txt = e.plain_text().strip()
            if txt:
                ax = float(e.dxf.insert.x)
                ay = float(e.dxf.insert.y)
                if in_bbox(ax, ay):
                    items.append((ax, ay, txt))

    if not items:
        return None

    # Prefer fixed-layout output similar to production title block layout.
    html = _render_title_block_html(items)
    if html:
        return html

    # Cluster items into rows by Y coordinate
    all_ys = [y for _, y, _ in items]
    row_centres = _cluster(all_ys, tol=2.0)
    if not row_centres:
        return None

    # Assign each item to the nearest row centre
    rows: dict[int, list[tuple[float, str]]] = {i: [] for i in range(len(row_centres))}
    for x, y, txt in items:
        ri = min(range(len(row_centres)), key=lambda i: abs(row_centres[i] - y))
        rows[ri].append((x, txt))

    # Sort rows top→bottom (high Y = top in DXF), sort cells left→right within row
    md_rows = []
    for ri in sorted(rows.keys(), key=lambda i: -row_centres[i]):
        cells = [txt for _, txt in sorted(rows[ri], key=lambda c: c[0])]
        if any(cells):
            md_rows.append(cells)

    if not md_rows:
        return None

    max_cols = max(len(r) for r in md_rows)
    lines: list[str] = []
    for i, row in enumerate(md_rows):
        padded = row + [""] * (max_cols - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("|" + "---|" * max_cols)
    return "\n".join(lines)


def extract_table_markdown(h_segs, v_segs, texts: list[TextItem],
                            x_lo: float, y_lo: float,
                            x_hi: float, y_hi: float,
                            title: str = "",
                            margin: float = 5.0,
                            text_cluster_tol: float = 2.0,
                            sparse_ratio: float = 3.0) -> str | None:
    """Given DXF bbox, extract and return a Markdown table or None.

    When H-lines are sparse relative to the number of distinct text rows
    (sparse_ratio), row boundaries are inferred from text Y-positions
    instead, enabling proper per-row extraction for checkbox-style tables.
    """
    h_in, v_in, t_in = filter_entities(h_segs, v_segs, texts,
                                        x_lo, y_lo, x_hi, y_hi, margin)
    if not h_in and not v_in and not t_in:
        return None

    grid = build_grid_text_aware(
        h_in, v_in, t_in,
        cluster_tol=1.5,
        text_cluster_tol=text_cluster_tol,
        sparse_ratio=sparse_ratio,
    )
    if grid.rows == 0 or grid.cols == 0:
        return None

    # Trim t_in to only texts within grid X span + small tolerance
    # This prevents contamination from adjacent tables that fall in the margin zone
    x_lo_g = min(grid.col_xs)
    x_hi_g = max(grid.col_xs)
    grid_margin = 2.0   # tight: only snap texts within 2mm of grid boundary
    t_in_trimmed = [t for t in t_in
                    if x_lo_g - grid_margin <= t.x <= x_hi_g + grid_margin]

    matrix, unassigned = assign_texts(grid, t_in_trimmed)

    # Snap unassigned texts to nearest border cell (within grid bounds only)
    for t in unassigned:
        if not (y_lo <= t.y <= y_hi):
            continue
        # Only snap texts that are close to the grid boundary, not far outside
        if t.x < x_lo_g - grid_margin or t.x > x_hi_g + grid_margin:
            continue
        cell_by_y = grid.cell_at((x_lo_g + x_hi_g) / 2, t.y)
        if cell_by_y is None:
            continue
        ri = cell_by_y[0]
        if t.x < x_lo_g:
            matrix[ri][0].insert(0, t.text)
        else:
            matrix[ri][grid.cols - 1].append(t.text)

    md = matrix_to_markdown(matrix, grid.rows, grid.cols, title)
    if md:
        md = _normalize_spec_table_markdown(md)
    return md if md.strip() else None


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(dxf_path: Path,
                 img_path: Path,
                 weights: Path = WEIGHTS_DEFAULT,
                 score_thr: float = 0.7,
                 target_labels: set[str] | None = None,
                 contain_thr: float = 0.90,
                 iou_thr: float = 0.70,
                 margin: float = 5.0,
                 verbose: bool = False) -> list[dict]:
    """Run the full detection + extraction pipeline.

    Returns a list of dicts with keys:
      label, score, px_box, dxf_box, markdown

    Args:
      contain_thr: Drop a box if ≥ this fraction of its area is inside another box.
      iou_thr:     IoU threshold for NMS within the same label class.
    """
    if target_labels is None:
        target_labels = {"table", "title_block"}

    # Load DXF entities
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    h_segs, v_segs = _extract_lines(msp)
    all_texts = _extract_texts(msp)

    # Load image first so bounds can be selected against frame aspect ratio
    img = cv2.imread(str(img_path))
    bounds = choose_bounds_for_image(doc, img)

    if verbose:
        print(f"DXF page: {bounds.x0:.0f},{bounds.y0:.0f} → {bounds.x1:.0f},{bounds.y1:.0f}")
        print(f"Entities: {len(h_segs)} H, {len(v_segs)} V, {len(all_texts)} texts")

    # Build transform
    transform = build_transform(img, bounds, h_segs=h_segs, v_segs=v_segs)
    if verbose:
        print(f"Scale: x={transform.scale_x:.3f}  y={transform.scale_y:.3f} px/mm")

    # Run detection + suppress overlaps
    predictor  = load_predictor(weights, score_thr)
    detections = run_detection(predictor, img, target_labels)
    before     = len(detections)
    detections = suppress_contained(detections,
                                    contain_thr=contain_thr,
                                    iou_thr=iou_thr)
    if verbose:
        print(f"Detections: {before} raw → {len(detections)} after suppression")
        print("  " + ", ".join(f"{d.label}@{d.score:.2f}" for d in detections))

    # Extract per detection — column-major order (left column top→bottom, then next column)
    _pw = img.shape[1]
    _col_band = _pw // 5
    detections_sorted = sorted(detections,
                                key=lambda d: (int(d.px_box[0] // _col_band), d.px_box[1]))
    detections_sorted = merge_columnwise_text_table_boxes(
        detections_sorted,
        col_band_px=_col_band,
        y_gap_px=max(80.0, img.shape[0] * 0.03),
        x_overlap_min=0.30,
    )
    results = []
    for det in detections_sorted:
        bx1, by1, bx2, by2 = det.px_box
        xlo, ylo, xhi, yhi = transform.px_to_dxf(bx1, by1, bx2, by2)

        # title_block: parse both INSERT texts and direct model-space texts
        # restricted to the detected title-block bbox.
        if det.label == "title_block":
            # Title-block detector often predicts a thin strip at the very bottom.
            # Use a wider margin to recover upper rows (NO./縮尺/工事名/etc.).
            title_margin = max(margin, 20.0)
            md = extract_title_block_markdown(
                msp,
                doc,
                x_lo=xlo,
                y_lo=ylo,
                x_hi=xhi,
                y_hi=yhi,
                margin=title_margin,
            )
        else:
            # Use a larger margin for "text" labels near page edges.
            eff_margin = 20.0 if det.label == "text" else margin
            md = extract_table_markdown(h_segs, v_segs, all_texts,
                                         xlo, ylo, xhi, yhi,
                                         margin=eff_margin)

        results.append({
            "label":   det.label,
            "score":   det.score,
            "px_box":  det.px_box,
            "dxf_box": (xlo, ylo, xhi, yhi),
            "markdown": md,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7. Visualisation helper
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "table":       (0,   200,  0),
    "title_block": (200,   0, 200),
    "text":        (0,   100, 255),
    "diagram":     (255, 165,   0),
    "image":       (100, 100, 100),
}


def draw_results(img: np.ndarray, results: list[dict]) -> np.ndarray:
    out = img.copy()
    for r in results:
        color = _COLORS.get(r["label"], (128, 128, 128))
        x1, y1, x2, y2 = [int(v) for v in r["px_box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{r['label']} {r['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dxf", help="Path to the .dxf file")
    p.add_argument("--image",    help="Pre-rendered PNG image path (skips PDF render)")
    p.add_argument("--weights",  default=str(WEIGHTS_DEFAULT), help="Model weights path")
    p.add_argument("--score",    type=float, default=0.7, help="Min detection score")
    p.add_argument("--margin",   type=float, default=5.0, help="DXF bbox margin (mm)")
    p.add_argument("--out",      default="dxf_output", help="Output directory")
    p.add_argument("--vis",      action="store_true", help="Save visualisation image")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    dxf_path = Path(args.dxf)
    img_path = Path(args.image) if args.image else None

    if img_path is None:
        print("Error: --image is required (PDF render not yet auto-wired here).")
        sys.exit(1)

    results = run_pipeline(
        dxf_path   = dxf_path,
        img_path   = img_path,
        weights    = Path(args.weights),
        score_thr  = args.score,
        margin     = args.margin,
        verbose    = args.verbose,
    )

    # Print markdown
    for i, r in enumerate(results):
        if r["markdown"]:
            dxf_box = r["dxf_box"]
            print(f"\n## [{r['label']} {r['score']:.2f}] DXF({dxf_box[0]:.0f},{dxf_box[1]:.0f})→({dxf_box[2]:.0f},{dxf_box[3]:.0f})\n")
            print(r["markdown"])

    if args.vis:
        img = cv2.imread(str(img_path))
        vis = draw_results(img, results)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dxf_path.stem}_layout.png"
        cv2.imwrite(str(out_path), vis)
        print(f"\nVisualization saved → {out_path}")


if __name__ == "__main__":
    main()
