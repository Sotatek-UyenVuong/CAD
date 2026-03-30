"""
test_surya.py
-------------
Run Surya layout detection on CAD test images and compare with YOLO results.

Usage:
  uv run python tools/test_surya.py
  uv run python tools/test_surya.py --n 10 --out runs/surya_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Surya label → our label mapping
SURYA_TO_OURS: dict[str, str | None] = {
    "Text":         "text",
    "TextInlineMath": "text",
    "Caption":      "text",
    "Footnote":     "text",
    "PageFooter":   "text",
    "PageHeader":   "text",
    "SectionHeader": "text",
    "Title":        "text",
    "Table":        "table",
    "Figure":       "diagram",
    "FigureGroup":  "diagram",
    "TableGroup":   "table",
    "Form":         "table",
    "Handwriting":  None,
    "Picture":      None,
    "Code":         None,
    "Formula":      None,
    "ListItem":     "text",
}

LABEL_COLORS = {
    "text":    (0,   128, 255),
    "table":   (0,   200, 0),
    "diagram": (200, 0,   200),
    "other":   (128, 128, 128),
}


def draw_boxes(img, predictions: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for pred in predictions:
        bbox  = pred["bbox"]   # [x1, y1, x2, y2]
        label = pred["label"]
        color = LABEL_COLORS.get(label, LABEL_COLORS["other"])
        draw.rectangle(bbox, outline=color, width=3)
        draw.text((bbox[0] + 4, bbox[1] + 4), label, fill=color, font=font)

    img.save(out_path)


def run(args: argparse.Namespace) -> None:
    try:
        from surya.layout import LayoutPredictor  # type: ignore
    except ImportError:
        import sys
        sys.exit("ERROR: surya-ocr not installed.\n  uv pip install surya-ocr --no-deps")

    from PIL import Image

    test_dir = ROOT / "dataset" / "images" / "test"
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_paths = sorted(test_dir.glob("*.png"))[: args.n]
    if not images_paths:
        import sys
        sys.exit(f"ERROR: No images found in {test_dir}")

    print(f"▶  Loading Surya layout model …")
    from surya.layout import FoundationPredictor  # type: ignore
    foundation = FoundationPredictor(device="cuda")
    predictor  = LayoutPredictor(foundation)
    print(f"▶  Running on {len(images_paths)} images …\n")

    pil_images = [Image.open(p).convert("RGB") for p in images_paths]
    layout_results = predictor(pil_images)

    label_counts: dict[str, int] = {}
    for idx, (img_path, result, pil_img) in enumerate(zip(images_paths, layout_results, pil_images)):
        predictions = []
        for block in result.bboxes:
            surya_label = block.label
            our_label   = SURYA_TO_OURS.get(surya_label, "other")
            if our_label is None:
                continue
            predictions.append({
                "bbox":  [int(v) for v in block.bbox],
                "label": our_label,
            })
            label_counts[our_label] = label_counts.get(our_label, 0) + 1

        out_path = out_dir / f"{img_path.stem}_surya.jpg"
        draw_boxes(pil_img.copy(), predictions, out_path)
        print(f"  [{idx+1:>3}/{len(images_paths)}] {img_path.name}  →  {len(predictions)} detections  →  {out_path.name}")

    print(f"\n✓ Saved {len(images_paths)} annotated images to: {out_dir}")
    print(f"\n  Detection counts (mapped labels):")
    for label, cnt in sorted(label_counts.items()):
        print(f"    {label:<12} : {cnt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Surya layout detection on CAD test images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",   type=int, default=20, help="Number of test images to process")
    parser.add_argument("--out", type=str, default="runs/surya_eval", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
