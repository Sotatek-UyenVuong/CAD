"""
detect_drawings_crop.py
────────────────────────────────────────────────────────────
Workflow:
  1. Gemini phân tích ảnh low-res → detect bbox từng bản vẽ (tọa độ 0-100 %)
  2. Render PDF trang đó ở DPI cao (base_dpi × zoom, mặc định 6×)
  3. Scale bbox → crop từng bản vẽ từ ảnh high-res
  4. Lưu crop + JSON + ảnh annotated

Usage:
  python detect_drawings_crop.py <pdf> <page_png> [options]
  python detect_drawings_crop.py pdf/超高層集合住宅samescale.pdf \
      rendered_超高層集合住宅samescale_dpi300/page_0.png \
      --zoom 6 --page 0 --out crops_超高層
"""

import argparse
import json
import os
import re
import sys

import fitz                          # PyMuPDF
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

load_dotenv()

# ── Gemini setup ─────────────────────────────────────────────────────────────

def setup_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')


# ── Step 1: Gemini detect drawings ───────────────────────────────────────────

DETECT_PROMPT = """
You are analyzing a Japanese architectural/CAD document image that contains
MULTIPLE SEPARATE FLOOR PLAN DRAWINGS arranged on a single page (like a catalog sheet).

Your task: detect the bounding box of EACH individual floor plan drawing.

**WHAT COUNTS AS ONE DRAWING:**
- A distinct floor plan with its own title (building name, floor info, scale, area stats)
- Each drawing is visually separated from others (whitespace or thin border)
- Include the title text block that belongs to that drawing (name, RC造, etc.)
- Include the north arrow (方位記号) if it's inside the drawing area
- Do NOT merge multiple drawings into one box

**WHAT TO EXCLUDE from each box:**
- Page header / overall document title at the very top (e.g. "030-01 基準階事例...")
- Page-wide legend or footer
- Whitespace between drawings

**OUTPUT - Return ONLY valid JSON, no markdown fences:**
{
  "drawings": [
    {
      "index": 0,
      "title": "1. MMタワーズ  RC造 30F/B1F",
      "x_min": 2.1,
      "y_min": 5.0,
      "x_max": 34.5,
      "y_max": 51.0
    },
    ...
  ]
}

Coordinates are PERCENTAGES of image width (x) and height (y), range 0.0–100.0.
Be precise. Return ONLY the JSON.
"""


def detect_drawings_gemini(model, image_path: str, debug: bool = False) -> list[dict]:
    """
    Returns list of drawing dicts:
      {index, title, x_min, y_min, x_max, y_max}  (coords in 0-100 %)
    """
    print(f"🔍 Gemini analyzing: {image_path}")
    pil_image = Image.open(image_path)
    w, h = pil_image.size
    print(f"   Image size: {w}×{h}")

    response = model.generate_content([DETECT_PROMPT, pil_image])
    raw = response.text.strip()

    if debug:
        print("── Gemini raw response ──")
        print(raw)
        print("────────────────────────")

    # Strip markdown fences if present
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```$', '', raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        drawings = data.get('drawings', [])
        print(f"   ✅ Detected {len(drawings)} drawings")
        for d in drawings:
            print(f"      [{d['index']}] {d.get('title', '')}  "
                  f"({d['x_min']:.1f},{d['y_min']:.1f}) → ({d['x_max']:.1f},{d['y_max']:.1f})")
        return drawings
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print("Raw response:")
        print(raw)
        return []


# ── Step 2+3: Render & crop each drawing directly via PyMuPDF clip ───────────
# Không load toàn trang high-res vào RAM (tránh PIL DecompressionBombError).
# Thay vào đó dùng fitz clip để render từng vùng nhỏ ở DPI cao.

def crop_drawings_from_pdf(
    drawings: list[dict],
    pdf_path: str,
    page_num: int,
    target_dpi: int,
    out_dir: str,
    padding_pct: float = 0.5,
) -> list[dict]:
    """
    Với mỗi drawing bbox (tọa độ % từ Gemini):
      1. Chuyển % → tọa độ PDF points
      2. Dùng fitz.clip để render chỉ vùng đó ở target_dpi
      3. Lưu PNG riêng

    Không cần load toàn trang vào RAM → tránh DecompressionBombError.
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_rect = page.rect  # kích thước trang tính bằng PDF points

    zoom = target_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    W_pt = page_rect.width
    H_pt = page_rect.height
    pad_x_pt = W_pt * padding_pct / 100
    pad_y_pt = H_pt * padding_pct / 100

    print(f"\n✂️  Cropping {len(drawings)} drawings from PDF @ {target_dpi} DPI "
          f"(page size: {W_pt:.0f}×{H_pt:.0f} pt)")

    results = []
    for d in drawings:
        idx = d['index']
        title = d.get('title', f'drawing_{idx}')

        # % → PDF points (với padding)
        x1 = max(0.0, d['x_min'] / 100 * W_pt - pad_x_pt)
        y1 = max(0.0, d['y_min'] / 100 * H_pt - pad_y_pt)
        x2 = min(W_pt, d['x_max'] / 100 * W_pt + pad_x_pt)
        y2 = min(H_pt, d['y_max'] / 100 * H_pt + pad_y_pt)

        clip = fitz.Rect(x1, y1, x2, y2)
        pix = page.get_pixmap(matrix=mat, clip=clip)

        safe_title = re.sub(r'[^\w\-]', '_', title)[:40]
        crop_path = os.path.join(out_dir, f"drawing_{idx:02d}_{safe_title}.png")
        pix.save(crop_path)

        info = {
            **d,
            'crop_path': crop_path,
            'crop_w': pix.width,
            'crop_h': pix.height,
            'clip_pt': [x1, y1, x2, y2],
        }
        results.append(info)
        print(f"   [{idx:02d}] {pix.width}×{pix.height} px → {os.path.basename(crop_path)}")

    return results


# ── Step 4: Annotated overview ────────────────────────────────────────────────

def save_annotated(drawings: list[dict], lowres_path: str, out_dir: str):
    """Draw bounding boxes on the low-res image for a quick overview."""
    img = Image.open(lowres_path).convert('RGB')
    W, H = img.size
    draw = ImageDraw.Draw(img)

    colors = ['#FF3B30', '#FF9500', '#34C759', '#007AFF',
              '#AF52DE', '#FF2D55', '#5AC8FA', '#FFCC00']

    # Try to load a font
    font = None
    for fp in [
        '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        '/System/Library/Fonts/Helvetica.ttc',
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, max(12, H // 60))
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    for d in drawings:
        idx = d['index']
        color = colors[idx % len(colors)]
        x1 = d['x_min'] / 100 * W
        y1 = d['y_min'] / 100 * H
        x2 = d['x_max'] / 100 * W
        y2 = d['y_max'] / 100 * H

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"[{idx}] {d.get('title', '')}"
        draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)

    ann_path = os.path.join(out_dir, '_annotated_overview.png')
    img.save(ann_path)
    print(f"\n📌 Annotated overview → {ann_path}")
    return ann_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Detect & crop individual drawings from a multi-plan PDF page using Gemini.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python detect_drawings_crop.py \\\n'
            '      pdf/超高層集合住宅samescale.pdf \\\n'
            '      rendered_超高層集合住宅samescale_dpi300/page_0.png\n\n'
            '  python detect_drawings_crop.py pdf/foo.pdf rendered_foo_dpi300/page_0.png \\\n'
            '      --zoom 6 --page 0 --out my_crops --padding 1.0 --debug\n'
        )
    )
    parser.add_argument('pdf',      help='PDF file path')
    parser.add_argument('page_png', help='Existing low-res PNG of the page (used for Gemini detection)')
    parser.add_argument('--page',    type=int, default=0,
                        help='Page number (0-indexed, default: 0)')
    parser.add_argument('--zoom',    type=float, default=6.0,
                        help='Zoom factor vs base 300 DPI (default: 6 → 1800 DPI)')
    parser.add_argument('--base-dpi', type=int, default=300,
                        help='DPI of the existing low-res PNG (default: 300)')
    parser.add_argument('--out',     default=None,
                        help='Output directory for crops (default: crops_<pdf_stem>_p<page>)')
    parser.add_argument('--padding', type=float, default=0.5,
                        help='Extra padding around each crop in %% of image dim (default: 0.5)')
    parser.add_argument('--debug',   action='store_true',
                        help='Print raw Gemini response')
    parser.add_argument('--skip-render', action='store_true',
                        help='Skip PDF rendering (reuse existing high-res PNG if present)')
    parser.add_argument('--json-only', action='store_true',
                        help='Only run Gemini detection, skip rendering and cropping')

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"❌ PDF not found: {args.pdf}")
        sys.exit(1)
    if not os.path.exists(args.page_png):
        print(f"❌ Page PNG not found: {args.page_png}")
        sys.exit(1)

    pdf_stem = os.path.splitext(os.path.basename(args.pdf))[0]
    out_dir = args.out or f"crops_{pdf_stem}_p{args.page}"

    target_dpi = int(args.base_dpi * args.zoom)

    print("=" * 60)
    print("DETECT DRAWINGS & CROP (Gemini + PyMuPDF)")
    print("=" * 60)
    print(f"PDF:        {args.pdf}")
    print(f"Page PNG:   {args.page_png}  (base {args.base_dpi} DPI)")
    print(f"Page:       {args.page}")
    print(f"Zoom:       {args.zoom}× → target {target_dpi} DPI")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    # Step 1 — Gemini detection
    model = setup_gemini()
    drawings = detect_drawings_gemini(model, args.page_png, debug=args.debug)

    if not drawings:
        print("❌ No drawings detected. Exiting.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Save detection JSON
    json_path = os.path.join(out_dir, 'detections.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(drawings, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Detections saved → {json_path}")

    if args.json_only:
        save_annotated(drawings, args.page_png, out_dir)
        print("\n✅ Done (--json-only mode)")
        return

    # Step 2+3 — Render + crop each drawing directly (no full-page load)
    print(f"\n✂️  Rendering each drawing @ {target_dpi} DPI via PDF clip …")
    results = crop_drawings_from_pdf(
        drawings, args.pdf, args.page, target_dpi, out_dir,
        padding_pct=args.padding
    )

    # Step 4 — Annotated overview on low-res
    save_annotated(drawings, args.page_png, out_dir)

    # Save full results JSON
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pdf': args.pdf,
            'page': args.page,
            'base_dpi': args.base_dpi,
            'target_dpi': target_dpi,
            'zoom': args.zoom,
            'drawings': results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Full results → {results_path}")
    print(f"\n✅ Done! {len(results)} drawings cropped → {out_dir}/")
    print("\nFiles created:")
    for r in results:
        print(f"  {r['crop_path']}  ({r['crop_w']}×{r['crop_h']} px)")


if __name__ == '__main__':
    main()
