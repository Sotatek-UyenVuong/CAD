"""
test_layoutparser.py
--------------------
Run LayoutParser (PaddleDetection backend, PubLayNet model) on CAD test images.

Usage:
  uv run python tools/test_layoutparser.py
  uv run python tools/test_layoutparser.py --n 20 --out runs/layoutparser_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# PubLayNet labels → our labels
PUBLAYNET_TO_OURS: dict[str, str | None] = {
    "Text":    "text",
    "Title":   "text",
    "List":    "text",
    "Table":   "table",
    "Figure":  "diagram",
}

LABEL_COLORS = {
    "text":    (0,   128, 255),
    "table":   (0,   200, 0),
    "diagram": (200, 0,   200),
    "other":   (128, 128, 128),
}


def draw_boxes(img, predictions: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for pred in predictions:
        bbox  = pred["bbox"]
        label = pred["label"]
        score = pred.get("score", 1.0)
        color = LABEL_COLORS.get(label, LABEL_COLORS["other"])
        draw.rectangle(bbox, outline=color, width=3)
        draw.text((bbox[0] + 4, bbox[1] + 4), f"{label} {score:.2f}", fill=color, font=font)

    img.save(out_path)


def run(args: argparse.Namespace) -> None:
    import layoutparser as lp
    from PIL import Image
    import numpy as np

    test_dir = ROOT / "dataset" / "images" / "test"
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_paths = sorted(test_dir.glob("*.png"))[: args.n]
    if not images_paths:
        import sys
        sys.exit(f"ERROR: No images found in {test_dir}")

    print(f"▶  Loading LayoutParser PaddleDetection model (PubLayNet) …")
    model = lp.PaddleDetectionLayoutModel(
        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        enforce_cpu=False,
        extra_config={"threshold": args.conf},
    )
    print(f"▶  Running on {len(images_paths)} images …\n")

    label_counts: dict[str, int] = {}
    for idx, img_path in enumerate(images_paths):
        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_pil)

        layout = model.detect(img_np)

        predictions = []
        for block in layout:
            lp_label  = block.type
            our_label = PUBLAYNET_TO_OURS.get(lp_label, "other")
            x1, y1, x2, y2 = (int(v) for v in block.block.coordinates)
            predictions.append({
                "bbox":  [x1, y1, x2, y2],
                "label": our_label,
                "score": round(block.score, 2) if hasattr(block, "score") else 1.0,
            })
            label_counts[our_label] = label_counts.get(our_label, 0) + 1

        out_path = out_dir / f"{img_path.stem}_lp.jpg"
        draw_boxes(img_pil.copy(), predictions, out_path)
        print(f"  [{idx+1:>3}/{len(images_paths)}] {img_path.name}  →  {len(predictions)} detections  →  {out_path.name}")

    print(f"\n✓ Saved {len(images_paths)} annotated images to: {out_dir}")
    print(f"\n  Detection counts (mapped labels):")
    for label, cnt in sorted(label_counts.items()):
        print(f"    {label:<12} : {cnt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test LayoutParser (PubLayNet) on CAD test images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",    type=int,   default=20,                     help="Number of test images")
    parser.add_argument("--out",  type=str,   default="runs/layoutparser_eval", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3,                    help="Confidence threshold")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
