"""grid_engine.py — Rule-based CAD grid detection + R-tree spatial indexing.

Pipeline:
  1. Extract H/V lines (+ LWPOLYLINE segments)
  2. Cluster similar coordinates → canonical row/col boundaries
  3. Build R-tree index over cells
  4. Assign each TEXT to its cell in O(n log n)
  5. Output structured grid → list[list[str]]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import ezdxf
from rtree import index as rtree_index


# ──────────────────────────────────────────────────────────────────────────────
# Snap / cluster helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cluster(values: list[float], tol: float) -> list[float]:
    """Merge values within *tol* of each other → sorted unique cluster centres."""
    if not values:
        return []
    vals = sorted(values)
    clusters: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _snap(value: float, grid: list[float]) -> float:
    """Return the nearest value in *grid*."""
    return min(grid, key=lambda g: abs(g - value))


# ──────────────────────────────────────────────────────────────────────────────
# Geometry extraction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextItem:
    text:   str
    x:      float
    y:      float
    height: float
    layer:  str


def _extract_lines(msp) -> tuple[list[tuple[float,float,float,float]],
                                  list[tuple[float,float,float,float]]]:
    """Return (h_lines, v_lines) where each entry is (x_lo, y, x_hi, y) / (x, y_lo, x, y_hi)."""
    TOL = 1.0          # max deviation to call a line "horizontal" or "vertical"

    h_segs: list[tuple[float,float,float,float]] = []
    v_segs: list[tuple[float,float,float,float]] = []

    def _classify(x1: float, y1: float, x2: float, y2: float) -> None:
        if abs(y1 - y2) <= TOL:                          # horizontal
            h_segs.append((min(x1,x2), (y1+y2)/2, max(x1,x2), (y1+y2)/2))
        elif abs(x1 - x2) <= TOL:                        # vertical
            v_segs.append(((x1+x2)/2, min(y1,y2), (x1+x2)/2, max(y1,y2)))

    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            _classify(e.dxf.start.x, e.dxf.start.y, e.dxf.end.x, e.dxf.end.y)
        elif t == "LWPOLYLINE":
            pts = list(e.get_points("xy"))
            if e.is_closed:
                pts.append(pts[0])
            for i in range(len(pts) - 1):
                _classify(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

    return h_segs, v_segs


def _extract_texts(msp) -> list[TextItem]:
    items: list[TextItem] = []
    for e in msp:
        t = e.dxftype()
        if t == "TEXT":
            txt = e.dxf.text.strip()
            if txt:
                items.append(TextItem(txt,
                    float(e.dxf.insert.x), float(e.dxf.insert.y),
                    float(e.dxf.height), e.dxf.layer))
        elif t == "MTEXT":
            txt = e.plain_text().strip()
            if txt:
                items.append(TextItem(txt,
                    float(e.dxf.insert.x), float(e.dxf.insert.y),
                    float(e.dxf.char_height), e.dxf.layer))
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Grid detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Grid:
    """Detected grid with row/col boundaries and R-tree index over cells."""
    row_ys:   list[float]   # sorted Y positions of horizontal lines (bottom→top)
    col_xs:   list[float]   # sorted X positions of vertical lines (left→right)
    # Derived
    rows:     int = field(init=False)
    cols:     int = field(init=False)
    _idx:     object = field(init=False, repr=False)
    _cells:   list[tuple[int,int]] = field(init=False, repr=False)  # (ri, ci) per R-tree id

    def __post_init__(self):
        self.rows = max(0, len(self.row_ys) - 1)
        self.cols = max(0, len(self.col_xs) - 1)
        self._build_rtree()

    def _build_rtree(self):
        """Index each cell rectangle in the R-tree."""
        self._idx   = rtree_index.Index()
        self._cells = []
        cell_id = 0
        for ri in range(self.rows):
            y_lo = self.row_ys[ri]
            y_hi = self.row_ys[ri + 1]
            for ci in range(self.cols):
                x_lo = self.col_xs[ci]
                x_hi = self.col_xs[ci + 1]
                # rtree bbox: (left, bottom, right, top)
                self._idx.insert(cell_id, (x_lo, y_lo, x_hi, y_hi))
                self._cells.append((ri, ci))
                cell_id += 1

    def cell_at(self, x: float, y: float) -> tuple[int,int] | None:
        """Return (row_idx, col_idx) for point (x,y), or None if outside grid."""
        hits = list(self._idx.intersection((x, y, x, y)))
        if not hits:
            return None
        # Pick the tightest (smallest area) containing cell
        best = min(hits, key=lambda h: self._cell_area(h))
        return self._cells[best]

    def _cell_area(self, cell_id: int) -> float:
        ri, ci = self._cells[cell_id]
        return ((self.col_xs[ci+1] - self.col_xs[ci]) *
                (self.row_ys[ri+1] - self.row_ys[ri]))


def build_grid(h_segs: list, v_segs: list,
               cluster_tol: float = 1.5) -> Grid:
    """Cluster line positions → canonical boundaries → Grid."""
    raw_ys = [float(y) for _, y, _, _ in h_segs]
    raw_xs = [float(x) for x, _, _, _ in v_segs]

    row_ys = sorted(_cluster(raw_ys, cluster_tol))
    col_xs = sorted(_cluster(raw_xs, cluster_tol))

    return Grid(row_ys=row_ys, col_xs=col_xs)


# ──────────────────────────────────────────────────────────────────────────────
# Text → cell assignment
# ──────────────────────────────────────────────────────────────────────────────

def assign_texts(grid: Grid,
                 texts: list[TextItem]) -> list[list[list[str]]]:
    """Return matrix[row][col] = list of text strings in that cell.

    Row 0 = BOTTOM row (as stored in DXF; flip if needed for display).
    """
    matrix: list[list[list[str]]] = [
        [[] for _ in range(grid.cols)]
        for _ in range(grid.rows)
    ]
    unassigned: list[TextItem] = []

    for item in texts:
        cell = grid.cell_at(item.x, item.y)
        if cell is not None:
            ri, ci = cell
            matrix[ri][ci].append(item.text)
        else:
            unassigned.append(item)

    return matrix, unassigned


def snap_circles_to_labels(
    circle_cells: dict[tuple[int,int], bool],
    matrix: list[list[list[str]]],
    label_cols: list[int],
    section_row_range: tuple[int, int],
) -> dict[tuple[int,int], tuple[int,int]]:
    """Map circle cells to the nearest labeled row within section range.

    circle_cells : {(mat_ri, ci): True, ...}   — circles in matrix coordinates
    label_cols   : column indices that hold row labels (e.g. [16, 24])
    section_row_range : (start_mat_ri, end_mat_ri) in matrix coordinates

    Returns {(mat_ri_circle, ci): (mat_ri_label, ci)} mapping each circle
    to its best-matching labeled row.  If already in a labeled row, maps
    to itself.
    """
    start_ri, end_ri = section_row_range
    # Pre-build: for each label_col, sorted list of mat_ri that have a label
    labeled_rows: dict[int, list[int]] = {}
    for lc in label_cols:
        rows = [r for r in range(start_ri, end_ri)
                if r < len(matrix) and lc < len(matrix[r]) and matrix[r][lc]]
        labeled_rows[lc] = sorted(rows)

    result: dict[tuple[int,int], tuple[int,int]] = {}
    for (mat_ri, ci) in circle_cells:
        if not (start_ri <= mat_ri < end_ri):
            result[(mat_ri, ci)] = (mat_ri, ci)
            continue
        # Find nearest label_col for this ci
        best_lc = min(label_cols, key=lambda lc: abs(lc - ci)) if label_cols else ci
        rows = labeled_rows.get(best_lc, [])
        if not rows:
            result[(mat_ri, ci)] = (mat_ri, ci)
            continue
        # Already has a label in this row?
        if mat_ri in rows:
            result[(mat_ri, ci)] = (mat_ri, ci)
        else:
            # Snap to nearest labeled row
            nearest = min(rows, key=lambda r: abs(r - mat_ri))
            result[(mat_ri, ci)] = (nearest, ci)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_grid_from_dxf(
    dxf_path: str | Path,
    cluster_tol: float = 1.5,
) -> tuple[Grid, list[list[list[str]]], list[TextItem]]:
    """Full pipeline: DXF → (grid, matrix, unassigned_texts).

    matrix[row][col] = list[str]   (row 0 = bottom of drawing)
    """
    doc  = ezdxf.readfile(str(dxf_path))
    msp  = doc.modelspace()
    h_segs, v_segs = _extract_lines(msp)
    texts           = _extract_texts(msp)
    grid            = build_grid(h_segs, v_segs, cluster_tol)
    matrix, unassigned = assign_texts(grid, texts)
    return grid, matrix, unassigned


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python grid_engine.py <file.dxf>")
        sys.exit(1)

    grid, matrix, unassigned = parse_grid_from_dxf(path)

    print(f"Grid: {grid.rows} rows × {grid.cols} cols")
    print(f"Unassigned texts: {len(unassigned)}")
    print()

    # Print top 20 rows (display order = top to bottom → reverse)
    display = list(reversed(matrix))
    for ri, row in enumerate(display[:20]):
        cells = [" | ".join(c) if c else "" for c in row]
        print(f"  [{ri:3d}] {' │ '.join(cells[:8])}")
    if grid.rows > 20:
        print(f"  ... +{grid.rows - 20} more rows")
