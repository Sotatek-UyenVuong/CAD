"""
yolo_to_coco.py
---------------
Convert YOLO-format labels (dataset/labels/) to COCO JSON for Detectron2.
Writes train.json / val.json / test.json into coco_dataset/.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

CLASSES: list[str] = ["text", "table", "title_block", "diagram", "image"]

SPLITS = ["train", "val", "test"]

DATASET_DIR  = ROOT / "dataset"
COCO_OUT_DIR = ROOT / "coco_dataset"


def yolo_box_to_coco(cx: float, cy: float, w: float, h: float, W: int, H: int) -> list[float]:
    """Convert YOLO normalized cx cy w h → COCO [x_min, y_min, w, h] in pixels."""
    x_min = (cx - w / 2) * W
    y_min = (cy - h / 2) * H
    return [x_min, y_min, w * W, h * H]


def convert_split(split: str) -> None:
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split

    if not img_dir.exists():
        print(f"⚠  Missing: {img_dir}")
        return

    images      : list[dict] = []
    annotations : list[dict] = []
    img_id  = 0
    ann_id  = 0

    for img_path in sorted(img_dir.glob("*.png")):
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"⚠  Could not open {img_path}: {e}")
            continue

        img_id += 1
        images.append({
            "id"        : img_id,
            "file_name" : img_path.name,
            "width"     : W,
            "height"    : H,
        })

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bbox = yolo_box_to_coco(cx, cy, bw, bh, W, H)
            area = bbox[2] * bbox[3]
            ann_id += 1
            annotations.append({
                "id"          : ann_id,
                "image_id"    : img_id,
                "category_id" : cls_id + 1,   # COCO categories are 1-indexed
                "bbox"        : [round(v, 2) for v in bbox],
                "area"        : round(area, 2),
                "iscrowd"     : 0,
            })

    categories = [{"id": i + 1, "name": name, "supercategory": "cad"} for i, name in enumerate(CLASSES)]

    coco = {"images": images, "annotations": annotations, "categories": categories}

    out_path = COCO_OUT_DIR / f"{split}.json"
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"✓ {split:5s}  images {len(images):>4}  annotations {len(annotations):>5}  → {out_path}")


def main() -> None:
    COCO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        convert_split(split)
    print("\nDone. COCO JSON files written to:", COCO_OUT_DIR)


if __name__ == "__main__":
    main()
