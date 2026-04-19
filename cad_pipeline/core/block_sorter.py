"""block_sorter.py — Reading-order sorting and grouping for layout blocks.

Reading order: column-major
  1. Detect columns via x-axis sweep: sort by x1, track running x2 max,
     start new column when next block's x1 > running_x2 + gap_tolerance.
  2. Sort columns left → right (by column's min x1).
  3. Within each column, sort blocks top → bottom (by y1).
  4. Group consecutive text/table blocks that are vertically adjacent
     within the same column into a single logical "run".

Typical CAD sheet layout handled:
  ┌─────────────────────┬──────────────────┐
  │  Main drawing area  │  Spec tables     │
  │  (diagrams)         │  (top-to-bottom) │
  │                     ├──────────────────┤
  │                     │  Title block     │
  └─────────────────────┴──────────────────┘

  → Column 1: diagrams (top→bottom)
  → Column 2: tables, then title_block (top→bottom)
"""

from __future__ import annotations

GROUPABLE = {"text", "table"}   # block types that can be merged into runs

# Blocks whose y1 values fall within the same Y_TIE_BUCKET are considered
# "same row" and are sorted left → right by x1 (not top → bottom).
Y_TIE_BUCKET = 50


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def sort_reading_order(blocks: list[dict]) -> list[dict]:
    """Return blocks sorted in reading order: left-column → right, top → bottom.

    Each returned block gets a `_col_id` key (int, 0-based) indicating which
    column it belongs to. This is used by group_text_table_runs to avoid
    merging blocks from adjacent columns.

    Blocks with no bbox are appended at the end in original order.
    """
    if len(blocks) <= 1:
        # Still tag single block
        for b in blocks:
            b["_col_id"] = 0
        return list(blocks)

    valid, no_bbox = [], []
    for b in blocks:
        if b.get("bbox") and len(b["bbox"]) == 4:
            valid.append(b)
        else:
            no_bbox.append(b)

    if not valid:
        return list(blocks)

    columns = _detect_columns(valid)
    # _detect_columns always appends full-width blocks as the LAST column.
    # Split that off so it survives the x-based sort and stays at the end.
    full_width_col = columns.pop() if _has_full_width_blocks(columns) else []

    # Sort normal columns left → right by leftmost x1
    columns.sort(key=lambda col: min(b["bbox"][0] for b in col))

    # Re-attach full-width at the very end
    if full_width_col:
        columns.append(full_width_col)

    # Within each column: sort by (y-bucket, x1).
    # Blocks whose y1 values fall within the same Y_TIE_BUCKET are in the
    # "same row" and are ordered left → right by x1.
    for col_id, col in enumerate(columns):
        col.sort(key=lambda b: (int(b["bbox"][1] / Y_TIE_BUCKET), b["bbox"][0]))
        for b in col:
            b["_col_id"] = col_id

    result = [b for col in columns for b in col]
    for b in no_bbox:
        b["_col_id"] = -1
    result.extend(no_bbox)
    return result


def _has_full_width_blocks(columns: list[list[dict]]) -> bool:
    """Return True if the last column consists of full-width blocks."""
    if not columns:
        return False
    last = columns[-1]
    return bool(last) and all(
        (b["bbox"][2] - b["bbox"][0]) / max(
            max(b2["bbox"][2] for col in columns for b2 in col) -
            min(b2["bbox"][0] for col in columns for b2 in col), 1
        ) >= _FULL_WIDTH_RATIO
        for b in last
    )


BlockGroup = list[dict]  # a run of logically adjacent blocks


def group_text_table_runs(
    sorted_blocks: list[dict],
    y_gap_threshold: float = 60.0,
) -> list[BlockGroup]:
    """Group consecutive text/table blocks that are vertically adjacent in the
    same column into a single logical run.

    Two adjacent blocks (already in reading order) are merged when ALL hold:
      - Both are text or table type
      - They have the same _col_id (set by sort_reading_order)
      - The vertical gap between them is ≤ y_gap_threshold pixels

    Non-groupable blocks (diagram, title_block, image) are always solo groups.
    Blocks without _col_id (e.g. not passed through sort_reading_order) fall
    back to x-overlap check.
    """
    if not sorted_blocks:
        return []

    groups: list[BlockGroup] = []
    current: BlockGroup = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        prev = current[-1]

        # Determine same-column: prefer _col_id tag, else x-overlap
        prev_col = prev.get("_col_id")
        this_col = block.get("_col_id")
        if prev_col is not None and this_col is not None:
            same_column = prev_col == this_col
        else:
            same_column = _x_overlaps(prev["bbox"], block["bbox"])

        # Blocks in the same y-bucket are side-by-side (same row) — never merge
        same_row = (
            int(prev["bbox"][1] / Y_TIE_BUCKET) == int(block["bbox"][1] / Y_TIE_BUCKET)
        )

        if (
            block.get("type") in GROUPABLE
            and prev.get("type") in GROUPABLE
            and same_column
            and not same_row
            and _y_gap(prev["bbox"], block["bbox"]) <= y_gap_threshold
        ):
            current.append(block)
        else:
            groups.append(current)
            current = [block]

    groups.append(current)
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# Column detection — x-center gap analysis
# ══════════════════════════════════════════════════════════════════════════════

_FULL_WIDTH_RATIO = 0.75   # blocks spanning > 75 % of page width → full-width
_MIN_COL_GAP_PX  = 50.0    # minimum gap between column centers to split


def _detect_columns(blocks: list[dict]) -> list[list[dict]]:
    """Partition blocks into columns using x-center gap analysis.

    Algorithm:
      1. Full-width blocks (span > FULL_WIDTH_RATIO of page) are stripped out
         and appended as a final pseudo-column (sorted by y) so they appear
         at the end regardless of x-position.
      2. For remaining blocks, compute x-center for each block.
      3. Sort by x-center. Find gaps between consecutive centers.
      4. A gap is a "column break" when it is both:
           - > MIN_COL_GAP_PX pixels, AND
           - > 1.5 × median of all gaps   (avoids splitting within dense grids)
      5. Blocks on the same side of every break form one column.

    This correctly handles:
      - Standard 2-column layouts (wide x-gap between columns)
      - Dense table grids (many small local gaps, few large column gaps)
      - Full-width elements like title blocks at the bottom
    """
    if not blocks:
        return []

    # ── 1. Separate full-width blocks ────────────────────────────────────────
    all_x1 = [b["bbox"][0] for b in blocks]
    all_x2 = [b["bbox"][2] for b in blocks]
    page_x1 = min(all_x1)
    page_x2 = max(all_x2)
    page_width = page_x2 - page_x1 or 1.0

    normal, full_width = [], []
    for b in blocks:
        span = b["bbox"][2] - b["bbox"][0]
        if span / page_width >= _FULL_WIDTH_RATIO:
            full_width.append(b)
        else:
            normal.append(b)

    if not normal:
        return [full_width] if full_width else []

    # ── 2. Sort by x-center ───────────────────────────────────────────────────
    def x_ctr(b: dict) -> float:
        return (b["bbox"][0] + b["bbox"][2]) / 2.0

    sorted_by_ctr = sorted(normal, key=x_ctr)
    centers = [x_ctr(b) for b in sorted_by_ctr]

    # ── 3. Compute gaps between consecutive centers ───────────────────────────
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

    # ── 4. Find column-break threshold ───────────────────────────────────────
    if gaps:
        median_gap = sorted(gaps)[len(gaps) // 2]
        break_threshold = max(_MIN_COL_GAP_PX, median_gap * 1.5)
    else:
        break_threshold = _MIN_COL_GAP_PX

    # ── 5. Split into column groups ───────────────────────────────────────────
    columns: list[list[dict]] = [[sorted_by_ctr[0]]]
    for i, gap in enumerate(gaps):
        if gap >= break_threshold:
            columns.append([sorted_by_ctr[i + 1]])
        else:
            columns[-1].append(sorted_by_ctr[i + 1])

    # Append full-width blocks as a final group (sorted top→bottom)
    if full_width:
        full_width.sort(key=lambda b: b["bbox"][1])
        columns.append(full_width)

    return columns


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _x_overlaps(bbox_a: list, bbox_b: list, tolerance: float = 20.0) -> bool:
    """True if the two bboxes share x-range (i.e. are in the same column band)."""
    ax1, _, ax2, _ = bbox_a
    bx1, _, bx2, _ = bbox_b
    return ax1 <= bx2 + tolerance and bx1 <= ax2 + tolerance


def _y_gap(bbox_above: list, bbox_below: list) -> float:
    """Vertical gap in pixels between bottom of upper block and top of lower block."""
    y2_above = bbox_above[3]
    y1_below = bbox_below[1]
    return max(0.0, y1_below - y2_above)
