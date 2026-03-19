"""
split_page_tables.py
Tách trang bản vẽ CAD / 特記仕様書 thành các block dùng Gemini Vision.

Hai chế độ:
  --mode grid   (mặc định): Gemini trả vị trí đường kẻ ngang+dọc → lưới ô đều
  --mode auto              : Gemini trả bounding box từng block → linh hoạt
                             (block dọc cắt dọc, block ngang cắt ngang)

Usage:
  python split_page_tables.py <image_path> [options]

Examples:
  # Auto (linh hoạt, phù hợp trang hỗn hợp)
  python split_page_tables.py page_17.png --mode auto

  # Grid cột dọc (cho 特記仕様書 thuần text)
  python split_page_tables.py page_2.png --mode grid --cols-only --trim-header 0.04 --trim-footer 0.03

  # Auto + OCR tiêu đề
  python split_page_tables.py page_17.png --mode auto --ocr
"""

import io
import os
import re
import sys
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

MODEL_SPLIT = "gemini-3.1-pro-preview"
MODEL_OCR   = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# Gemini helpers
# ─────────────────────────────────────────────
_client = None

def _client_get():
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            sys.exit("❌ Cần đặt GEMINI_API_KEY trong .env")
        _client = genai.Client(api_key=key)
    return _client

def _call(model: str, img_bytes: bytes, prompt: str) -> str:
    from google.genai import types
    part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    resp = _client_get().models.generate_content(
        model=model,
        contents=[part, prompt],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=8192),
    )
    return resp.text.strip()

def _encode(img: Image.Image, max_dim: int = 2000) -> bytes:
    W, H = img.size
    scale = min(max_dim / W, max_dim / H, 1.0)
    if scale < 1.0:
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _parse_floats(text: str) -> list[float]:
    text = re.sub(r"```[a-z]*", "", text)
    try:
        arr = json.loads(text.strip())
        if isinstance(arr, list):
            return [float(v) for v in arr]
    except Exception:
        pass
    return [float(v) for v in re.findall(r"\d+\.?\d*", text)]

# ─────────────────────────────────────────────
# Detect split positions via Gemini
# ─────────────────────────────────────────────
_PROMPT_COLS = """\
This is a Japanese architectural drawing page (特記仕様書 / specification sheet).
The page is divided into MAJOR vertical columns by thick ruled lines.

Identify the X position of each major vertical dividing line as a fraction [0.0–1.0] of image width.
Include 0.0 (left edge) and 1.0 (right edge).
Ignore thin internal lines — only thick borders between major sections.

Return ONLY a JSON array of floats sorted left→right.
Example: [0.0, 0.25, 0.50, 0.75, 1.0]"""

_PROMPT_ROWS = """\
This is a Japanese architectural drawing page (特記仕様書 / specification sheet).
The page is divided into MAJOR horizontal sections by thick ruled lines.

Identify the Y position of each major horizontal dividing line as a fraction [0.0–1.0] of image height.
Include 0.0 (top edge) and 1.0 (bottom edge).
Ignore thin internal lines — only thick borders between major sections.

Return ONLY a JSON array of floats sorted top→bottom.
Example: [0.0, 0.05, 0.92, 1.0]"""

_PROMPT_AUTO = """\
This is a Japanese architectural drawing page. It may contain:
- Text specification sections arranged in multiple vertical columns
- Diagram/table sections arranged in horizontal rows
- Mixed layouts where each horizontal band has its own column structure

Task: identify every distinct rectangular block/section visible on this page.
Blocks are separated by thick ruled lines or clear whitespace.

For each block return:
{
  "id": <1-based integer>,
  "title": "<short semantic label describing what this block IS, in Japanese. Examples: 図面表題欄, 凡例, 仕様表, 建具表, 部屋名称表, 断面図, 平面図, 詳細図, 注記欄, スケール表示, 索引図. Do NOT copy raw text verbatim — summarise the PURPOSE of the block in 2–6 characters.>",
  "type": "<text/diagram/table/mixed>",
  "x_min": <float 0.0–1.0, left edge / image width>,
  "y_min": <float 0.0–1.0, top edge / image height>,
  "x_max": <float 0.0–1.0, right edge / image width>,
  "y_max": <float 0.0–1.0, bottom edge / image height>
}

Rules:
- Use fractions [0.0–1.0] relative to full image size.
- COVERAGE IS MANDATORY: together the blocks must cover the ENTIRE content area of the page.
  Do NOT leave any content region un-boxed. Every row of content must belong to a block.
- Include ALL distinct blocks — text columns, diagram panels, legend tables, title rows, footer rows, etc.
  Even narrow header/footer strips that span the full width must be included as blocks.
- Do NOT include the overall page outer border as a block.
- Blocks must NOT overlap. Every pixel on the page belongs to AT MOST one block.
  If two regions share content, assign it to exactly one block — do not duplicate.
- Typical count for this type of page: 15–40 blocks.

Return ONLY the JSON array, nothing else."""


def detect_splits_grid(img: Image.Image,
                       cols: bool = True,
                       rows: bool = True) -> tuple[list[int], list[int]]:
    """Grid mode: ask for H and V line positions separately."""
    W, H = img.size
    data = _encode(img)

    if cols:
        print(f"  📡 [{MODEL_SPLIT}] Phát hiện cột dọc...")
        v_fracs = sorted(set(_parse_floats(_call(MODEL_SPLIT, data, _PROMPT_COLS))))
        v_fracs = [f for f in v_fracs if 0.0 <= f <= 1.0]
        print(f"     → {v_fracs}")
    else:
        v_fracs = [0.0, 1.0]

    if rows:
        print(f"  📡 [{MODEL_SPLIT}] Phát hiện hàng ngang...")
        h_fracs = sorted(set(_parse_floats(_call(MODEL_SPLIT, data, _PROMPT_ROWS))))
        h_fracs = [f for f in h_fracs if 0.0 <= f <= 1.0]
        print(f"     → {h_fracs}")
    else:
        h_fracs = [0.0, 1.0]

    v_px = sorted(set(max(0, min(W, int(f * W))) for f in v_fracs))
    h_px = sorted(set(max(0, min(H, int(f * H))) for f in h_fracs))
    for lst, limit in [(v_px, W), (h_px, H)]:
        if not lst or lst[0] > 10:        lst.insert(0, 0)
        if not lst or lst[-1] < limit-10: lst.append(limit)
    return h_px, v_px


def detect_blocks_auto(img: Image.Image) -> list[dict]:
    """Auto mode: ask Gemini to return each block's bounding box directly."""
    W, H = img.size
    data = _encode(img)
    print(f"  📡 [{MODEL_SPLIT}] Phát hiện blocks tự động...")
    raw = _call(MODEL_SPLIT, data, _PROMPT_AUTO)
    raw = re.sub(r"```[a-z]*\s*", "", raw)
    raw = re.sub(r"```", "", raw).strip()

    try:
        blocks = json.loads(raw)
        if not isinstance(blocks, list):
            blocks = []
    except Exception as e:
        print(f"  ⚠️  JSON parse lỗi: {e} — dùng regex fallback")
        # regex fallback
        pat = re.compile(
            r'"id"\s*:\s*(\d+).*?"title"\s*:\s*"([^"]*)".*?'
            r'"x_min"\s*:\s*([\d.]+).*?"y_min"\s*:\s*([\d.]+).*?'
            r'"x_max"\s*:\s*([\d.]+).*?"y_max"\s*:\s*([\d.]+)',
            re.DOTALL,
        )
        blocks = [{"id": int(m[0]), "title": m[1], "type": "",
                   "x_min": float(m[2]), "y_min": float(m[3]),
                   "x_max": float(m[4]), "y_max": float(m[5])}
                  for m in pat.findall(raw)]

    # Convert fractions → pixels (dùng .get với fallback để tránh KeyError)
    cells = []
    for b in blocks:
        try:
            x0 = max(0, int(b.get("x_min", 0) * W))
            y0 = max(0, int(b.get("y_min", 0) * H))
            x1 = min(W, int(b.get("x_max", 1) * W))
            y1 = min(H, int(b.get("y_max", 1) * H))
        except (TypeError, ValueError):
            continue
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        cells.append({"id": b.get("id", len(cells) + 1), "row": 0, "col": len(cells),
                      "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "title": b.get("title", ""),
                      "type": b.get("type", "")})

    print(f"     → {len(cells)} blocks phát hiện được")
    cells = _remove_overlaps(cells, page_w=W, page_h=H)
    print(f"     → {len(cells)} blocks sau khi lọc overlap")
    return cells


def _iou(a: dict, b: dict) -> float:
    """True IoU — intersection / union. Safe against container blocks."""
    ix0 = max(a["x0"], b["x0"]); iy0 = max(a["y0"], b["y0"])
    ix1 = min(a["x1"], b["x1"]); iy1 = min(a["y1"], b["y1"])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    area_b = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _remove_overlaps(cells: list[dict], iou_thresh: float = 0.3,
                     page_w: int = 1, page_h: int = 1) -> list[dict]:
    """Loại bỏ block trùng lặp; loại block container chiếm > 70% diện tích trang."""
    page_area = page_w * page_h

    def area(c): return (c["x1"] - c["x0"]) * (c["y1"] - c["y0"])

    # Xóa block quá lớn (likely the outer border mistakenly included)
    filtered = [c for c in cells if area(c) / page_area < 0.70]
    if not filtered:
        filtered = cells  # fallback: giữ hết nếu tất cả đều lớn

    kept = []
    for cell in sorted(filtered, key=area, reverse=True):  # lớn trước
        if all(_iou(cell, k) < iou_thresh for k in kept):
            kept.append(cell)
    kept.sort(key=lambda c: c["id"])
    return kept



# ─────────────────────────────────────────────
# Build + save cells
# ─────────────────────────────────────────────
PALETTE = [(220,20,60),(0,160,0),(0,80,220),(200,120,0),
           (130,0,200),(0,170,170),(180,60,0),(0,130,90)]

def _font(size: int):
    for p in ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
              "/Library/Fonts/Arial Unicode MS.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

def _cells_from_grid(h_lines: list[int], v_lines: list[int],
                     min_w: int, min_h: int) -> list[dict]:
    cells, idx = [], 1
    for i in range(len(h_lines) - 1):
        y0, y1 = h_lines[i], h_lines[i + 1]
        if y1 - y0 < min_h: continue
        for j in range(len(v_lines) - 1):
            x0, x1 = v_lines[j], v_lines[j + 1]
            if x1 - x0 < min_w: continue
            cells.append({"id": idx, "row": i, "col": j,
                          "x0": x0, "y0": y0, "x1": x1, "y1": y1, "title": ""})
            idx += 1
    return cells


def save_cells(cells: list[dict],
               img: Image.Image, stem: str, out_dir: Path,
               ocr: bool) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cells:
        print("❌ Không có ô nào — thử --mode auto hoặc giảm --min-w / --min-h")
        return []

    # OCR titles
    if ocr:
        print(f"  🔤 OCR tiêu đề {len(cells)} ô...")
        for cell in cells:
            strip_h = max(20, int((cell["y1"] - cell["y0"]) * 0.15))
            crop = img.crop((cell["x0"], cell["y0"], cell["x1"], cell["y0"] + strip_h))
            prompt = ("Extract the title/label text from this Japanese drawing cell header. "
                      "Return ONLY the text (1–10 words). If unclear, return empty string.")
            try:   cell["title"] = _call(MODEL_OCR, _encode(crop, 800), prompt)
            except: cell["title"] = ""

    # Overview
    ov = img.copy().convert("RGB")
    draw = ImageDraw.Draw(ov)
    font = _font(28)
    for cell in cells:
        c = PALETTE[cell["id"] % len(PALETTE)]
        draw.rectangle([cell["x0"], cell["y0"], cell["x1"], cell["y1"]], outline=c, width=6)
        lbl = f"{cell['id']}" + (f" {cell['title'][:20]}" if cell["title"] else "")
        draw.text((cell["x0"] + 8, cell["y0"] + 6), lbl, fill=c, font=font)
    ov.save(out_dir / f"{stem}_overview.png")
    print(f"  📌 Overview → {out_dir / f'{stem}_overview.png'}")

    # Crop + save
    results = []
    for cell in cells:
        crop = img.crop((cell["x0"], cell["y0"], cell["x1"], cell["y1"]))
        safe = re.sub(r'[\\/:*?"<>|\s]+', "_", cell["title"])[:30] if cell["title"] else ""
        fname = f"{stem}_r{cell['row']:02d}c{cell['col']:02d}_id{cell['id']:03d}"
        if safe: fname += f"_{safe}"
        fpath = out_dir / (fname + ".png")
        crop.save(fpath)
        results.append({"id": cell["id"], "row": cell["row"], "col": cell["col"],
                        "title": cell["title"],
                        "type": cell.get("type", ""),
                        "bbox": [cell["x0"], cell["y0"], cell["x1"], cell["y1"]],
                        "size": [cell["x1"] - cell["x0"], cell["y1"] - cell["y0"]],
                        "file": str(fpath)})
        print(f"  ✂️  id={cell['id']:2d} [{cell['row']},{cell['col']}] "
              f"{cell['x1']-cell['x0']}×{cell['y1']-cell['y0']}px"
              + (f"  {cell['title'][:40]}" if cell["title"] else ""))

    manifest = out_dir / f"{stem}_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  📄 Manifest → {manifest}")
    return results

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["grid", "auto"], default="grid",
                    help="grid: lưới đều (tốt cho thuần text); auto: Gemini detect từng block (tốt cho trang hỗn hợp)")
    ap.add_argument("--cols-only", action="store_true",
                    help="[grid] Chỉ cắt cột dọc, không cắt hàng ngang")
    ap.add_argument("--min-w", type=int, default=40,
                    help="[grid] Min cell width in px (default 40)")
    ap.add_argument("--min-h", type=int, default=40,
                    help="[grid] Min cell height in px (default 40)")
    ap.add_argument("--trim-header", type=float, default=0.0,
                    help="Fraction of height to skip at top (e.g. 0.04)")
    ap.add_argument("--trim-footer", type=float, default=0.0,
                    help="Fraction of height to skip at bottom (e.g. 0.03)")
    ap.add_argument("--ocr", action="store_true",
                    help="Dùng Gemini OCR tiêu đề từng ô sau khi cắt")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <image_dir>/sections/)")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"❌ Không tìm thấy: {args.image}")

    img     = Image.open(args.image).convert("RGB")
    W, H    = img.size
    stem    = Path(args.image).stem
    out_dir = Path(args.out or (Path(args.image).parent / "sections"))

    print(f"\n🔍 {args.image}  ({W}×{H}px)  mode={args.mode}")

    # Trim header/footer
    y0 = int(H * args.trim_header)
    y1 = int(H * (1.0 - args.trim_footer)) if args.trim_footer else H
    img_work = img.crop((0, y0, W, y1)) if (y0 or y1 < H) else img

    print("📐 Phát hiện vị trí cắt (Gemini)...")

    if args.mode == "auto":
        cells = detect_blocks_auto(img_work)
        for c in cells:
            c["y0"] += y0; c["y1"] += y0
    else:
        h_lines, v_lines = detect_splits_grid(
            img_work, cols=True, rows=not args.cols_only)
        # Offset h_lines về ảnh gốc
        h_lines = [y + y0 for y in h_lines]
        if h_lines[0]  > y0 + 5: h_lines.insert(0, y0)
        if h_lines[-1] < y1 - 5: h_lines.append(y1)
        print(f"   {len(h_lines)-1} hàng × {len(v_lines)-1} cột"
              f"  → tối đa {(len(h_lines)-1)*(len(v_lines)-1)} ô")
        cells = _cells_from_grid(h_lines, v_lines, args.min_w, args.min_h)
        print(f"   ✅ {len(cells)} ô (w≥{args.min_w}, h≥{args.min_h})")

    print(f"\n✂️  Lưu → {out_dir}/")
    results = save_cells(cells, img, stem, out_dir, ocr=args.ocr)
    print(f"\n✅ Xong! {len(results)} blocks → {out_dir}/")

if __name__ == "__main__":
    main()
