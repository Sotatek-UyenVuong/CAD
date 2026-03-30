#!/usr/bin/env python3
"""yolo_dxf_mapper.py — Map YOLO bounding boxes to DXF entities.

Coordinate chain:
  YOLO (normalized 0–1) → PNG pixel → DXF model space

Usage:
  python yolo_dxf_mapper.py <yolo.txt> <file.dxf> <image.png> [--dpi 300]
  python yolo_dxf_mapper.py page_0.txt D000.dxf page_0.png
  python yolo_dxf_mapper.py page_0.txt D000.dxf page_0.png --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import ezdxf
from PIL import Image

# ─── Table grid reconstruction ───────────────────────────────────────────────

def _cluster(values: list[float], gap: float = 2.0) -> list[float]:
    """Merge nearby values into single representative (midpoint of cluster)."""
    if not values:
        return []
    vals = sorted(set(round(v, 1) for v in values))
    clusters: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def reconstruct_table(
    doc: ezdxf.document.Drawing,
    mx1: float, my1: float, mx2: float, my2: float,
    tol: float = 1.5,
) -> str | None:
    """
    Reconstruct a table from DXF grid lines and text entities in a region.
    Returns a Markdown table string, or None if grid cannot be determined.

    Algorithm:
      1. Collect all horizontal/vertical lines → row/col boundaries
      2. Cluster nearby lines → distinct grid positions
      3. For each cell, collect TEXT entities whose insert falls in cell bbox
      4. Mark CIRCLE presence (checkmark markers) per cell
      5. Render as Markdown
    """
    msp = doc.modelspace()

    # ── Collect grid lines ────────────────────────────────────────────────
    h_raw: list[float] = []
    # V-lines: track (x, span_length) — only keep full-height ones
    v_candidates: list[tuple[float, float]] = []

    region_h = my2 - my1
    region_w = mx2 - mx1
    # Minimum span to count as a real column boundary (30% of table height)
    min_v_span = region_h * 0.30

    def _in_region(x: float, y: float) -> bool:
        return mx1 - tol <= x <= mx2 + tol and my1 - tol <= y <= my2 + tol

    for e in msp:
        etype = e.dxftype()
        if etype == "LINE":
            s, end = e.dxf.start, e.dxf.end
            if not (_in_region(s.x, s.y) or _in_region(end.x, end.y)):
                continue
            dy = abs(end.y - s.y)
            dx = abs(end.x - s.x)
            if dy < tol and dx > tol:          # horizontal
                h_raw.append((s.y + end.y) / 2)
            elif dx < tol and dy > tol:         # vertical — record span
                v_candidates.append(((s.x + end.x) / 2, dy))

        elif etype == "LWPOLYLINE":
            pts = list(e.vertices())
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if not any(_in_region(x, y) for x, y in zip(xs, ys)):
                continue
            if max(ys) - min(ys) < tol:        # horizontal
                h_raw.append(sum(ys) / len(ys))
            if max(xs) - min(xs) < tol:        # vertical
                v_candidates.append((sum(xs) / len(xs), max(ys) - min(ys)))

    # Keep only long vertical lines (real column separators)
    # Group nearby X candidates, take max span per group
    v_all = sorted(v_candidates, key=lambda t: t[0])
    v_grouped: dict[float, float] = {}   # x_cluster → max_span
    for x, span in v_all:
        placed = False
        for cx in list(v_grouped):
            if abs(x - cx) <= tol * 3:
                # Merge into existing cluster (keep max span)
                new_cx = (cx * v_grouped[cx] + x * span) / (v_grouped[cx] + span + 1e-9)
                v_grouped[new_cx] = max(v_grouped.pop(cx), span)
                placed = True
                break
        if not placed:
            v_grouped[x] = span

    # Only keep verticals that span >= min_v_span
    v_raw = [x for x, span in v_grouped.items() if span >= min_v_span]

    # ── Cluster into distinct boundaries ─────────────────────────────────
    # Add region borders if missing
    h_raw += [my1, my2]
    v_raw += [mx1, mx2]

    row_edges = sorted(_cluster(h_raw, gap=tol), reverse=True)  # top→bottom
    col_edges = sorted(_cluster(v_raw, gap=tol))                 # left→right

    n_rows = len(row_edges) - 1
    n_cols = len(col_edges) - 1

    if n_rows < 1 or n_cols < 1:
        return None

    # ── Collect text & circles per cell ──────────────────────────────────
    # Build cell grid: cell[r][c] = list of strings
    grid: list[list[list[str]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    circle_grid: list[list[int]] = [[0]*n_cols for _ in range(n_rows)]

    def _find_cell(x: float, y: float) -> tuple[int, int] | None:
        """Return (row, col) index or None."""
        col = None
        for ci in range(n_cols):
            if col_edges[ci] - tol <= x <= col_edges[ci + 1] + tol:
                col = ci
                break
        row = None
        for ri in range(n_rows):
            # row_edges is top→bottom: row_edges[ri] >= row_edges[ri+1]
            if row_edges[ri + 1] - tol <= y <= row_edges[ri] + tol:
                row = ri
                break
        if row is None or col is None:
            return None
        return row, col

    for e in msp:
        etype = e.dxftype()
        if etype == "TEXT":
            pt = e.dxf.get("insert")
            if not pt or not _in_region(pt.x, pt.y):
                continue
            cell = _find_cell(pt.x, pt.y)
            if cell:
                txt = e.dxf.get("text", "").strip()
                if txt:
                    grid[cell[0]][cell[1]].append(txt)

        elif etype == "MTEXT":
            pt = e.dxf.get("insert")
            if not pt or not _in_region(pt.x, pt.y):
                continue
            cell = _find_cell(pt.x, pt.y)
            if cell:
                txt = e.plain_mtext().strip() if hasattr(e, "plain_mtext") else ""
                if txt:
                    grid[cell[0]][cell[1]].append(txt)

        elif etype == "CIRCLE":
            ctr = e.dxf.center
            if not _in_region(ctr.x, ctr.y):
                continue
            cell = _find_cell(ctr.x, ctr.y)
            if cell:
                circle_grid[cell[0]][cell[1]] += 1

    # Merge circles into grid (mark cells that have circles)
    for r in range(n_rows):
        for c in range(n_cols):
            if circle_grid[r][c] > 0 and not grid[r][c]:
                grid[r][c].append("●")

    # ── Find header row (first row with the most non-empty cells) ─────────
    # Determine which rows have content
    non_empty = [sum(1 for c in row if c) for row in grid]

    # ── Render Markdown table ─────────────────────────────────────────────
    def _cell_str(texts: list[str]) -> str:
        return " / ".join(t.replace("|", "｜").replace("\n", " ") for t in texts)

    md_lines: list[str] = []

    # Use first non-empty row as header
    header_row = 0
    for i, count in enumerate(non_empty):
        if count > 0:
            header_row = i
            break

    header = [_cell_str(grid[header_row][c]) or f"Col{c+1}" for c in range(n_cols)]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * n_cols) + "|")

    for r in range(n_rows):
        if r == header_row:
            continue
        row_cells = [_cell_str(grid[r][c]) for c in range(n_cols)]
        if any(row_cells):  # skip fully empty rows
            md_lines.append("| " + " | ".join(row_cells) + " |")

    # ── Remove marker-only columns (all ● or all empty) ──────────────────
    def _is_marker_col(c: int) -> bool:
        vals = {_cell_str(grid[r][c]) for r in range(n_rows)}
        vals.discard("")
        return vals <= {"●"}  # only ● or empty

    keep_cols = [c for c in range(n_cols) if not _is_marker_col(c)]

    if not keep_cols:
        return None

    # Re-render with kept columns only
    header = [_cell_str(grid[header_row][c]) or f"Col{c+1}" for c in keep_cols]
    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(keep_cols)) + "|")

    for r in range(n_rows):
        if r == header_row:
            continue
        row_cells = [_cell_str(grid[r][c]) for c in keep_cols]
        # Skip rows that are all empty or all ●
        non_empty = [v for v in row_cells if v and v != "●"]
        if non_empty:
            # Replace remaining ● with empty
            clean = [v if v != "●" else "" for v in row_cells]
            md_lines.append("| " + " | ".join(clean) + " |")

    if len(md_lines) <= 2:
        return None

    return (
        f"> Grid: {n_rows} rows × {len(keep_cols)} cols (filtered from {n_cols})\n\n"
        + "\n".join(md_lines)
    )


# ─── DXF entity types to extract ───────────────────────────────────────────
TEXT_TYPES  = {"TEXT", "MTEXT"}
BLOCK_TYPES = {"INSERT"}
LINE_TYPES  = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "SPLINE"}
DIM_TYPES   = {"DIMENSION"}

CLASS_NAMES = ["text", "table", "title_block", "diagram"]

# ─── Data classes ───────────────────────────────────────────────────────────
@dataclass
class YoloBBox:
    class_id: int
    class_name: str
    cx: float          # normalized 0-1
    cy: float
    w: float
    h: float

    # DXF model space bbox (computed)
    mx1: float = 0.0
    my1: float = 0.0
    mx2: float = 0.0
    my2: float = 0.0


@dataclass
class DxfEntity:
    type: str
    layer: str
    text: str | None = None
    x: float = 0.0
    y: float = 0.0
    block_name: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class RegionResult:
    bbox: YoloBBox
    texts: list[DxfEntity] = field(default_factory=list)
    blocks: list[DxfEntity] = field(default_factory=list)
    dimensions: list[DxfEntity] = field(default_factory=list)
    lines_count: int = 0
    layers: list[str] = field(default_factory=list)
    table_md: str | None = None   # reconstructed Markdown table (for TABLE class)


# ─── Coordinate mapping ─────────────────────────────────────────────────────
class CoordMapper:
    """Maps YOLO normalized coords ↔ DXF model space via viewport info."""

    def __init__(self, dxf_path: str, img_path: str):
        self.doc = ezdxf.readfile(dxf_path)
        self.img = Image.open(img_path)
        self.img_w, self.img_h = self.img.size

        self.model_x_min, self.model_x_max, \
        self.model_y_min, self.model_y_max = self._get_model_extents()

        print(f"Image: {self.img_w}×{self.img_h} px")
        print(f"Model extents: X[{self.model_x_min:.1f}, {self.model_x_max:.1f}]"
              f"  Y[{self.model_y_min:.1f}, {self.model_y_max:.1f}]")

    def _get_model_extents(self) -> tuple[float, float, float, float]:
        """Derive model space extents from EXTMIN/EXTMAX header, with viewport fallback."""
        # Prefer EXTMIN/EXTMAX — most reliable
        extmin = self.doc.header.get("$EXTMIN")
        extmax = self.doc.header.get("$EXTMAX")
        if extmin and extmax:
            # extmin/extmax may be Vec3 or tuple
            def _coord(v, i):
                return v[i] if isinstance(v, (tuple, list)) else getattr(v, "xyz"[i])
            x0, y0 = _coord(extmin, 0), _coord(extmin, 1)
            x1, y1 = _coord(extmax, 0), _coord(extmax, 1)
            if abs(x1 - x0) > 1 and abs(y1 - y0) > 1:
                return x0, x1, y0, y1

        # Fallback: pick the viewport whose view_center is inside the model extents
        best_vp = None
        for layout in self.doc.layouts:
            if layout.name == "Model":
                continue
            for e in layout:
                if e.dxftype() != "VIEWPORT":
                    continue
                vp = e.dxf
                if vp.get("width", 0) < 10:
                    continue
                # Prefer the viewport whose height is closest to standard paper (594/297)
                vh = vp.get("view_height", 0)
                if best_vp is None or abs(vh - 594) < abs(best_vp[0] - 594):
                    best_vp = (vh, vp)

        if best_vp:
            _, vp = best_vp
            cx_m = vp.get("view_center_point").x
            cy_m = vp.get("view_center_point").y
            vw_m = vp.get("width")
            vh_m = vp.get("view_height")
            return cx_m - vw_m / 2, cx_m + vw_m / 2, cy_m - vh_m / 2, cy_m + vh_m / 2

        raise ValueError("Cannot determine model extents from DXF")

    def yolo_to_model(self, bbox: YoloBBox) -> tuple[float, float, float, float]:
        """Convert YOLO normalized bbox to DXF model space (mx1,my1,mx2,my2)."""
        # Normalized → pixel
        px1 = (bbox.cx - bbox.w / 2) * self.img_w
        py1 = (bbox.cy - bbox.h / 2) * self.img_h
        px2 = (bbox.cx + bbox.w / 2) * self.img_w
        py2 = (bbox.cy + bbox.h / 2) * self.img_h

        # Pixel → model (Y is flipped: PNG top=0, DXF bottom=y_min)
        rng_x = self.model_x_max - self.model_x_min
        rng_y = self.model_y_max - self.model_y_min

        mx1 = self.model_x_min + (px1 / self.img_w) * rng_x
        mx2 = self.model_x_min + (px2 / self.img_w) * rng_x
        my1 = self.model_y_max - (py2 / self.img_h) * rng_y   # py2→my1 (flip)
        my2 = self.model_y_max - (py1 / self.img_h) * rng_y   # py1→my2

        return mx1, my1, mx2, my2

    def model_to_norm(self, mx: float, my: float) -> tuple[float, float]:
        """Model space point → normalized image coordinates."""
        nx = (mx - self.model_x_min) / (self.model_x_max - self.model_x_min)
        ny = (self.model_y_max - my) / (self.model_y_max - self.model_y_min)
        return nx, ny


# ─── YOLO file parsing ───────────────────────────────────────────────────────
def load_yolo(yolo_path: str, class_names: list[str]) -> list[YoloBBox]:
    boxes = []
    for line in Path(yolo_path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        boxes.append(YoloBBox(
            class_id=cid, class_name=name,
            cx=float(parts[1]), cy=float(parts[2]),
            w=float(parts[3]), h=float(parts[4]),
        ))
    return boxes


# ─── DXF entity extraction ───────────────────────────────────────────────────
def _point_in_bbox(x: float, y: float,
                   mx1: float, my1: float, mx2: float, my2: float) -> bool:
    return mx1 <= x <= mx2 and my1 <= y <= my2


def extract_entities_in_region(
    doc: ezdxf.document.Drawing,
    mx1: float, my1: float, mx2: float, my2: float,
) -> tuple[list[DxfEntity], list[DxfEntity], list[DxfEntity], int]:
    msp = doc.modelspace()
    texts, blocks, dims = [], [], []
    lines_count = 0

    for e in msp:
        etype = e.dxftype()
        layer = e.dxf.get("layer", "")

        # ── Text entities ─────────────────────────────────────────────────
        if etype == "TEXT":
            pt = e.dxf.get("insert")
            if pt and _point_in_bbox(pt.x, pt.y, mx1, my1, mx2, my2):
                texts.append(DxfEntity(
                    type=etype, layer=layer,
                    text=e.dxf.get("text", "").strip(),
                    x=pt.x, y=pt.y,
                ))

        elif etype == "MTEXT":
            pt = e.dxf.get("insert")
            if pt and _point_in_bbox(pt.x, pt.y, mx1, my1, mx2, my2):
                texts.append(DxfEntity(
                    type=etype, layer=layer,
                    text=e.plain_mtext().strip() if hasattr(e, "plain_mtext") else "",
                    x=pt.x, y=pt.y,
                ))

        # ── Block inserts ──────────────────────────────────────────────────
        elif etype == "INSERT":
            pt = e.dxf.get("insert")
            if pt and _point_in_bbox(pt.x, pt.y, mx1, my1, mx2, my2):
                blocks.append(DxfEntity(
                    type=etype, layer=layer,
                    block_name=e.dxf.get("name", ""),
                    x=pt.x, y=pt.y,
                ))

        # ── Dimensions ────────────────────────────────────────────────────
        elif etype == "DIMENSION":
            pt = e.dxf.get("defpoint") or e.dxf.get("text_midpoint")
            if pt and _point_in_bbox(pt.x, pt.y, mx1, my1, mx2, my2):
                dims.append(DxfEntity(
                    type=etype, layer=layer,
                    text=str(e.dxf.get("actual_measurement", "")),
                    x=pt.x, y=pt.y,
                ))

        # ── Lines / geometry ──────────────────────────────────────────────
        elif etype == "LINE":
            s = e.dxf.get("start")
            if s and _point_in_bbox(s.x, s.y, mx1, my1, mx2, my2):
                lines_count += 1

        elif etype in {"LWPOLYLINE", "POLYLINE"}:
            try:
                pts = list(e.vertices())
                if any(_point_in_bbox(v[0], v[1], mx1, my1, mx2, my2) for v in pts):
                    lines_count += 1
            except Exception:
                pass

    return texts, blocks, dims, lines_count


# ─── Markdown export ─────────────────────────────────────────────────────────
def _write_markdown(results: list[RegionResult], out_path: str,
                    dxf_path: str, yolo_path: str) -> None:
    lines: list[str] = []

    lines.append(f"# YOLO → DXF Text Extract")
    lines.append(f"")
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| DXF  | `{Path(dxf_path).name}` |")
    lines.append(f"| YOLO | `{Path(yolo_path).name}` |")
    lines.append(f"| Boxes | {len(results)} |")
    lines.append(f"")

    # Group by class
    from collections import defaultdict
    by_class: dict[str, list[RegionResult]] = defaultdict(list)
    for r in results:
        by_class[r.bbox.class_name].append(r)

    for cls_name, cls_results in by_class.items():
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {cls_name.upper()} ({len(cls_results)} regions)")
        lines.append(f"")

        for idx, r in enumerate(cls_results):
            b = r.bbox
            lines.append(f"### Region {idx + 1}")
            lines.append(f"")
            lines.append(f"| Attr | Value |")
            lines.append(f"|------|-------|")
            lines.append(f"| YOLO center | ({b.cx:.4f}, {b.cy:.4f}) |")
            lines.append(f"| YOLO size   | {b.w:.4f} × {b.h:.4f} |")
            lines.append(f"| Model X     | [{b.mx1:.1f}, {b.mx2:.1f}] |")
            lines.append(f"| Model Y     | [{b.my1:.1f}, {b.my2:.1f}] |")
            lines.append(f"| Layers      | {', '.join(r.layers) if r.layers else '—'} |")
            lines.append(f"")

            # Table reconstruction (for TABLE class)
            if r.table_md:
                lines.append(r.table_md)
                lines.append(f"")
            else:
                # Fallback: plain text list
                all_texts = [t.text for t in r.texts if t.text and t.text.strip()]
                if all_texts:
                    lines.append(f"**Texts ({len(all_texts)}):**")
                    lines.append(f"")
                    for txt in all_texts:
                        safe = txt.replace("|", "｜").replace("\n", " ")
                        lines.append(f"- {safe}")
                    lines.append(f"")
                else:
                    lines.append(f"*No text found in this region.*")
                    lines.append(f"")

            # Blocks
            if r.blocks:
                block_names = [b.block_name for b in r.blocks if b.block_name]
                from collections import Counter
                counted = Counter(block_names)
                lines.append(f"**Blocks ({len(r.blocks)}):**")
                lines.append(f"")
                for name, cnt in counted.most_common(10):
                    lines.append(f"- `{name}` × {cnt}")
                lines.append(f"")

            # Dimensions
            if r.dimensions:
                lines.append(f"**Dimensions ({len(r.dimensions)}):**")
                lines.append(f"")
                for d in r.dimensions[:10]:
                    if d.text:
                        lines.append(f"- {d.text}")
                lines.append(f"")

            # Lines count
            if r.lines_count:
                lines.append(f"*Lines/polylines in region: {r.lines_count}*")
                lines.append(f"")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Map YOLO bboxes to DXF entities")
    ap.add_argument("yolo",  help="YOLO .txt file (normalized bboxes)")
    ap.add_argument("dxf",   help="DXF file")
    ap.add_argument("image", help="PNG image (rendered from same PDF)")
    ap.add_argument("--classes", default="",
                    help="Comma-separated class names (default: text,table,title_block,diagram)")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--md", action="store_true", help="Output Markdown file")
    ap.add_argument("--out", default="",
                    help="Output file path (default: auto-named from input)")
    ap.add_argument("--class-filter", default="",
                    help="Only process boxes of this class name")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.classes.split(",")] if args.classes else CLASS_NAMES

    # Load YOLO detections
    boxes = load_yolo(args.yolo, class_names)
    print(f"YOLO detections: {len(boxes)}")

    if args.class_filter:
        boxes = [b for b in boxes if b.class_name == args.class_filter]
        print(f"Filtered to '{args.class_filter}': {len(boxes)} boxes")

    # Build coordinate mapper
    mapper = CoordMapper(args.dxf, args.image)

    results: list[RegionResult] = []

    for i, bbox in enumerate(boxes):
        mx1, my1, mx2, my2 = mapper.yolo_to_model(bbox)
        bbox.mx1, bbox.my1, bbox.mx2, bbox.my2 = mx1, my1, mx2, my2

        texts, blocks, dims, lcount = extract_entities_in_region(
            mapper.doc, mx1, my1, mx2, my2
        )

        layers = list({e.layer for e in texts + blocks + dims})

        # Reconstruct table grid for TABLE class
        table_md = None
        if bbox.class_name == "table":
            table_md = reconstruct_table(mapper.doc, mx1, my1, mx2, my2)

        result = RegionResult(
            bbox=bbox,
            texts=texts,
            blocks=blocks,
            dimensions=dims,
            lines_count=lcount,
            layers=layers,
            table_md=table_md,
        )
        results.append(result)

        if not args.json and not args.md:
            print(f"\n{'─'*60}")
            print(f"[{i}] class={bbox.class_name} | "
                  f"YOLO({bbox.cx:.3f},{bbox.cy:.3f}) | "
                  f"model X[{mx1:.1f},{mx2:.1f}] Y[{my1:.1f},{my2:.1f}]")
            print(f"  Texts ({len(texts)}):")
            for t in texts[:10]:
                print(f"    [{t.layer}] '{t.text}'")
            if len(texts) > 10:
                print(f"    ... +{len(texts)-10} more")
            print(f"  Blocks ({len(blocks)}): {[b.block_name for b in blocks[:5]]}")
            print(f"  Dimensions: {len(dims)}  |  Lines: {lcount}")
            print(f"  Layers: {layers}")

    if args.json:
        def _serial(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)
        out_path = args.out or (Path(args.yolo).stem + "_dxf.json")
        Path(out_path).write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False,
                       indent=2, default=_serial),
            encoding="utf-8",
        )
        print(f"JSON saved → {out_path}")

    if args.md:
        out_path = args.out or (Path(args.yolo).stem + "_dxf.md")
        _write_markdown(results, out_path, args.dxf, args.yolo)
        print(f"Markdown saved → {out_path}")


if __name__ == "__main__":
    main()
