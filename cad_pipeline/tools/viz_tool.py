"""viz_tool.py — Draw bounding boxes on page images for count results.

Two modes:
  Mode 1 (DXF):    count result has "positions" with WCS coords
                   + viewport_bounds (xmin/xmax/ymin/ymax in WCS)
                   → linear WCS → pixel transform → draw boxes

  Mode 2 (Vision): count result has "positions" with normalized [0–1] coords
                   → scale to image size → draw boxes
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Font selection — prefer Noto (CJK+Latin) then DejaVu (Latin diacritics) ─
_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_LABEL_FONT_SIZE = 14
_HEADER_FONT_SIZE = 15


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# Colour palette for different labels (cycles if > 10 items)
_COLOURS = [
    (0, 200, 100),   # green
    (255, 80,  20),  # orange-red
    (255, 200,  0),  # yellow
    (0, 140, 255),   # blue
    (180,  0, 255),  # purple
    (0, 220, 220),   # cyan
    (255,  60, 160), # pink
    (120, 255,  60), # lime
    (255, 160,   0), # amber
    (60,  60, 255),  # royal blue
]


def _label_colour(label: str, idx: int) -> tuple[int, int, int]:
    return _COLOURS[idx % len(_COLOURS)]


# ── WCS → pixel helpers ──────────────────────────────────────────────────────

class ViewportTransform:
    """Linear transform from DXF model-space WCS to image pixel coords."""

    def __init__(
        self,
        wcs_xmin: float, wcs_xmax: float,
        wcs_ymin: float, wcs_ymax: float,
        img_w: int, img_h: int,
    ):
        self.wcs_xmin = wcs_xmin
        self.wcs_xmax = wcs_xmax
        self.wcs_ymin = wcs_ymin
        self.wcs_ymax = wcs_ymax
        self.img_w = img_w
        self.img_h = img_h

    def wcs_to_px(self, wcs_x: float, wcs_y: float) -> tuple[int, int]:
        px = int((wcs_x - self.wcs_xmin) / (self.wcs_xmax - self.wcs_xmin) * self.img_w)
        py = int((1.0 - (wcs_y - self.wcs_ymin) / (self.wcs_ymax - self.wcs_ymin)) * self.img_h)
        return px, py

    def box_size_px(self, wcs_size: float = 2000.0) -> int:
        """Convert a WCS size to approximate pixels (for default box radius)."""
        return max(20, int(wcs_size / (self.wcs_xmax - self.wcs_xmin) * self.img_w))


def get_viewport_bounds_from_dxf(dxf_path: str | Path, layout_name: str | None = None) -> dict | None:
    """Extract WCS bounds of the largest active viewport in the given layout.

    Args:
        dxf_path:    Path to .dxf file
        layout_name: Layout name substring to match (e.g. "地下1", "B1", "3F").
                     If None, uses the first non-Model layout.

    Returns:
        {x_min, x_max, y_min, y_max, layout} or None
    """
    try:
        import ezdxf  # type: ignore
    except ImportError:
        return None

    doc = ezdxf.readfile(str(dxf_path))
    target_layouts = []

    for layout in doc.layouts:
        if layout.name == "Model":
            continue
        if layout_name is None or layout_name.lower() in layout.name.lower():
            target_layouts.append(layout)

    if not target_layouts:
        return None

    # Pick best viewport across all matching layouts
    best_vp = None
    best_layout_name = ""
    for layout in target_layouts:
        for vp in layout.query("VIEWPORT"):
            if vp.dxf.status == 0:
                continue
            if best_vp is None or vp.dxf.view_height > best_vp.dxf.view_height:
                best_vp = vp
                best_layout_name = layout.name

    if best_vp is None:
        return None

    cx = best_vp.dxf.view_center_point.x + best_vp.dxf.view_target_point.x
    cy = best_vp.dxf.view_center_point.y + best_vp.dxf.view_target_point.y
    vh = best_vp.dxf.view_height
    vw = vh * (best_vp.dxf.width / best_vp.dxf.height)

    return {
        "layout": best_layout_name,
        "x_min": cx - vw / 2,
        "x_max": cx + vw / 2,
        "y_min": cy - vh / 2,
        "y_max": cy + vh / 2,
    }


# ── Core draw function ───────────────────────────────────────────────────────

def draw_count_boxes(
    image_path: str | Path,
    count_result: dict,
    output_path: str | Path | None = None,
    viewport_bounds: dict | None = None,
    box_wcs_size: float = 3000.0,
    box_norm_size: float = 0.02,
) -> Path:
    """Draw bounding boxes on a page image from a count_tool result.

    Args:
        image_path:       Source PNG/JPG image path
        count_result:     Result dict from run_count_tool (must have "positions")
        output_path:      Output image path. Defaults to <stem>_viz.<ext>
        viewport_bounds:  {x_min, x_max, y_min, y_max} in WCS.
                          Required for mode "dxf_*". Auto-derived from dxf if omitted.
        box_wcs_size:     Half-size of box in WCS units (for dxf modes)
        box_norm_size:    Half-size of box in normalized coords (for vision mode,
                          used when Gemini returns a point rather than a box)

    Returns:
        Path to the output image.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]
    mode = count_result.get("mode", "")
    positions = count_result.get("positions", [])
    query = count_result.get("query", "")
    count = count_result.get("count", 0)

    if not positions:
        pil_img = _cv2_to_pil(img)
        draw = ImageDraw.Draw(pil_img)
        _draw_header(draw, query, count, mode, img_w)
        img = _pil_to_cv2(pil_img)
        out = _save(img, image_path, output_path)
        return out

    is_dxf = mode.startswith("dxf")
    is_vision = mode == "vision_pro"

    # Build transform for DXF modes
    transform: ViewportTransform | None = None
    if is_dxf and viewport_bounds:
        transform = ViewportTransform(
            viewport_bounds["x_min"], viewport_bounds["x_max"],
            viewport_bounds["y_min"], viewport_bounds["y_max"],
            img_w, img_h,
        )

    # Assign colours per unique label
    unique_labels = list(dict.fromkeys(p["label"] for p in positions))
    label_colour_map = {lbl: _label_colour(lbl, i) for i, lbl in enumerate(unique_labels)}

    drawn: list[dict] = []  # for legend

    # Detect Gemini coordinate scale for vision mode (0-1, 0-100, or 0-1000)
    vision_scale = 100.0
    if is_vision and positions:
        all_vals = [
            v for p in positions
            for k in ("x_min", "x_max", "y_min", "y_max")
            if (v := p.get(k)) is not None and v > 1.0
        ]
        if all_vals and max(all_vals) > 100.0:
            vision_scale = 1000.0  # Gemini returned 0-1000 internal coords

    for pos in positions:
        label = pos.get("label", query)
        colour = label_colour_map.get(label, _COLOURS[0])

        if is_dxf and transform:
            wcs_x = pos.get("wcs_x", 0.0)
            wcs_y = pos.get("wcs_y", 0.0)
            cx_px, cy_px = transform.wcs_to_px(wcs_x, wcs_y)
            half = transform.box_size_px(box_wcs_size)
            x1, y1 = cx_px - half, cy_px - half
            x2, y2 = cx_px + half, cy_px + half

        elif is_vision:
            def _pct_to_px(v: float, dim: int, scale: float = 100.0) -> int:
                """Convert Gemini coordinate to pixel.
                Gemini uses 0-1 float, 0-100 percent, or 0-1000 internal coords.
                `scale` is pre-computed from the position set (100 or 1000)."""
                if v <= 1.0:
                    return int(v * dim)            # 0-1 normalized
                return int(v / scale * dim)        # scale to pixel

            if "x_min" in pos and "x_max" in pos:
                x1 = _pct_to_px(pos["x_min"], img_w, vision_scale)
                y1 = _pct_to_px(pos["y_min"], img_h, vision_scale)
                x2 = _pct_to_px(pos["x_max"], img_w, vision_scale)
                y2 = _pct_to_px(pos["y_max"], img_h, vision_scale)
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                min_box = max(20, int(0.015 * min(img_w, img_h)))
                if x2 - x1 < min_box:
                    mid = (x1 + x2) // 2
                    x1, x2 = mid - min_box // 2, mid + min_box // 2
                if y2 - y1 < min_box:
                    mid = (y1 + y2) // 2
                    y1, y2 = mid - min_box // 2, mid + min_box // 2
            else:
                cx_n = _pct_to_px(pos.get("cx", pos.get("x", 50)), img_w, vision_scale)
                cy_n = _pct_to_px(pos.get("cy", pos.get("y", 50)), img_h, vision_scale)
                half_px = int(box_norm_size * max(img_w, img_h))
                x1, y1 = cx_n - half_px, cy_n - half_px
                x2, y2 = cx_n + half_px, cy_n + half_px
        else:
            continue

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        # Draw filled rect (semi-transparent) + border using OpenCV
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        drawn.append({"label": label, "colour": colour, "x1": x1, "y1": y1})

    # ── Pillow pass: Unicode-safe text (labels + header) ──────────────────
    pil_img = _cv2_to_pil(img)
    draw = ImageDraw.Draw(pil_img)

    for item in drawn:
        _draw_label(draw, item["label"], item["x1"], item["y1"], item["colour"], img_h)

    _draw_header(draw, query, count, mode, img_w)

    img = _pil_to_cv2(pil_img)
    out = _save(img, image_path, output_path)
    return out


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
    img_h: int,
) -> None:
    """Draw Unicode-safe label text above a box using Pillow."""
    font = _load_font(_LABEL_FONT_SIZE)
    # Convert BGR colour → RGB for Pillow
    r, g, b = colour[2], colour[1], colour[0]
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x + 2
    ty = max(y - th - 4, 2)
    # Background rectangle
    draw.rectangle((tx - 2, ty - 1, tx + tw + 3, ty + th + 2), fill=(10, 10, 10))
    draw.text((tx, ty), text, font=font, fill=(r, g, b))


def _draw_header(
    draw: ImageDraw.ImageDraw,
    query: str,
    count: int,
    mode: str,
    img_w: int,
) -> None:
    """Draw Unicode-safe header bar at the top of the image."""
    font = _load_font(_HEADER_FONT_SIZE)
    header = f"Query: {query}  |  Count: {count}  |  Mode: {mode}"
    bbox = draw.textbbox((0, 0), header, font=font)
    th = bbox[3] - bbox[1]
    bar_h = th + 10
    draw.rectangle((0, 0, img_w, bar_h), fill=(20, 20, 20))
    draw.text((8, 5), header, font=font, fill=(255, 255, 255))


def _save(img: np.ndarray, src: Path, output_path: str | Path | None) -> Path:
    if output_path is None:
        output_path = src.parent / f"{src.stem}_viz{src.suffix}"
    output_path = Path(output_path)
    cv2.imwrite(str(output_path), img)
    return output_path
