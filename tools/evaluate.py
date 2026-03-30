"""
evaluate.py
-----------
Evaluate a trained YOLO model on the test split and optionally run inference
on individual images to visualise detections.

Usage:
  # Evaluate best.pt on test set
  python tools/evaluate.py --weights runs/detect/cad_yolo/weights/best.pt

  # Predict on a single image
  python tools/evaluate.py --weights runs/detect/cad_yolo/weights/best.pt \
      --predict dataset/images/test/folder0__page_5.png

  # Predict on an entire folder
  python tools/evaluate.py --weights runs/detect/cad_yolo/weights/best.pt \
      --predict dataset/images/test/ --conf 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "dataset" / "cad_dataset.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate or run inference with a trained YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", type=Path,
        default=ROOT / "runs" / "detect" / "cad_yolo" / "weights" / "best.pt",
        help="Path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--data", type=Path, default=DATASET_YAML,
        help="Dataset YAML (used for val/test evaluation)",
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Inference image size (must match training size)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device", default="0",
        help="CUDA device id or 'cpu'",
    )
    parser.add_argument(
        "--predict", type=Path, default=None,
        help="Optional: path to an image or folder for visual prediction (skips evaluation)",
    )
    parser.add_argument(
        "--save-txt", action="store_true",
        help="Save YOLO-format label .txt files alongside predictions",
    )
    parser.add_argument(
        "--project", type=Path, default=ROOT / "runs" / "eval",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--name", default="cad_eval",
        help="Run subdirectory name",
    )
    return parser.parse_args(argv)


def load_model(weights: Path):
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        sys.exit("ERROR: ultralytics not installed.\n  pip install ultralytics")

    if not weights.exists():
        sys.exit(
            f"ERROR: Weights not found: {weights}\n"
            f"  Run training first: python tools/train.py"
        )
    print(f"▶  Loading weights: {weights}")
    return YOLO(str(weights))


def run_evaluation(args: argparse.Namespace) -> None:
    model = load_model(args.weights)

    if not args.data.exists():
        sys.exit(
            f"ERROR: Dataset YAML not found: {args.data}\n"
            f"  Run: python tools/prepare_dataset.py"
        )

    print(f"▶  Evaluating on '{args.split}' split …\n")
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )

    print("\n─── Results ─────────────────────────────────────────────")
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")
    print("─────────────────────────────────────────────────────────")

    class_names = metrics.names
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print("\n  Per-class AP@0.5:")
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            print(f"    {class_names[int(idx)]:<15} {ap:.4f}")

    print(f"\n  Saved results to: {args.project / args.name}")


def run_prediction(args: argparse.Namespace) -> None:
    model = load_model(args.weights)
    source = args.predict

    if not source.exists():
        sys.exit(f"ERROR: Source not found: {source}")

    print(f"▶  Predicting on: {source}\n")
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print(f"\n✓ Predictions saved to: {args.project / args.name}")
    print(f"  Total images processed: {len(results)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.predict:
        run_prediction(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
