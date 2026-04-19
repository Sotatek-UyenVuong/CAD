"""
predict_layout.py
-----------------
Chạy layout detection trên ảnh bất kỳ (không cần trong dataset).

Usage:
  # Một ảnh
  /opt/conda/bin/python tools/predict_layout.py --input path/to/image.png

  # Nhiều ảnh / thư mục
  /opt/conda/bin/python tools/predict_layout.py --input path/to/folder/

  # Tùy chỉnh threshold
  /opt/conda/bin/python tools/predict_layout.py --input img.png --score-thr 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT    = Path(__file__).resolve().parent.parent.parent  # project root
_LD     = ROOT / "layout_detect"
WEIGHTS = _LD / "models" / "checkpoints" / "cad_layout_v7_swapsplit" / "model_final.pth"
OUT_DIR = _LD / "models" / "checkpoints" / "cad_layout_v7_swapsplit" / "predict"

CLASSES = ["text", "table", "title_block", "diagram"]
COLORS  = {0: (255, 80, 80), 1: (80, 200, 80), 2: (80, 80, 255), 3: (255, 165, 0)}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_cfg(score_thr: float):
    from detectron2.config import get_cfg          # type: ignore
    from detectron2.model_zoo import model_zoo     # type: ignore

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS                     = str(WEIGHTS)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(CLASSES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.INPUT.MIN_SIZE_TEST               = 1280
    cfg.INPUT.MAX_SIZE_TEST               = 2000
    cfg.freeze()
    return cfg


def draw_predictions(img, instances, score_thr: float):
    boxes  = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    labels = instances.pred_classes.cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[int(label)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        text = f"{CLASSES[label]} {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input",     required=True, help="Ảnh hoặc thư mục chứa ảnh")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--out-dir",   default=str(OUT_DIR))
    parser.add_argument("--save-json", action="store_true", help="Lưu kết quả JSON kèm ảnh")
    args = parser.parse_args()

    from detectron2.engine import DefaultPredictor  # type: ignore

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg       = build_cfg(args.score_thr)
    predictor = DefaultPredictor(cfg)

    images = collect_images(Path(args.input))
    if not images:
        print(f"Không tìm thấy ảnh nào trong: {args.input}")
        sys.exit(1)

    print(f"\n▶  Predicting {len(images)} image(s)  →  {out_dir}\n")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] không đọc được: {img_path}")
            continue

        outputs   = predictor(img)
        instances = outputs["instances"]

        # ── Vẽ bbox ─────────────────────────────────────────────────────────
        vis = draw_predictions(img.copy(), instances, args.score_thr)
        out_img = out_dir / img_path.name
        cv2.imwrite(str(out_img), vis)

        # ── In kết quả ra terminal ───────────────────────────────────────────
        boxes  = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        labels = instances.pred_classes.cpu().numpy()
        dets   = [
            {"class": CLASSES[int(l)], "score": float(s),
             "bbox": [float(v) for v in b]}
            for b, s, l in zip(boxes, scores, labels)
            if s >= args.score_thr
        ]
        print(f"  {img_path.name}  →  {len(dets)} detection(s)")
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            print(f"    [{d['class']:12s}] score={d['score']:.3f}  "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        # ── Lưu JSON ─────────────────────────────────────────────────────────
        if args.save_json:
            json_path = out_dir / (img_path.stem + ".json")
            json_path.write_text(json.dumps({"file": str(img_path), "detections": dets}, indent=2))

        print(f"    saved → {out_img}")

    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
