"""
train_detectron2.py
-------------------
Fine-tune Detectron2 Faster R-CNN (COCO pretrained) on CAD layout dataset.

Usage:
  uv run python tools/train_detectron2.py
  uv run python tools/train_detectron2.py --resume
  uv run python tools/train_detectron2.py --eval-only
  uv run python tools/train_detectron2.py --max-iter 30000 --lr 0.00025
  # Train trên train+val, đánh giá test, lưu thư mục riêng:
  uv run python tools/train_detectron2.py --merge-val --out-name cad_layout_trainval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent.parent  # project root
_LD       = ROOT / "layout_detect"                          # layout_detect/
# Dataset trên NVMe SSD để tăng tốc data loading (fallback về HDD nếu không có)
_SSD_DATA = Path("/tmp/cad_train")
COCO_DIR  = _SSD_DATA / "coco_dataset" if _SSD_DATA.exists() else _LD / "data" / "coco_dataset"
IMG_DIR   = _SSD_DATA / "dataset" / "images" if _SSD_DATA.exists() else _LD / "data" / "yolo_dataset" / "images"
WEIGHTS   = _LD / "models" / "pretrained" / "coco_faster_rcnn_R101_FPN_3x.pkl"
RUNS_DIR  = _LD / "models" / "checkpoints"

CLASSES  = ["text", "table", "title_block", "diagram", "image"]
NUM_CLASSES = len(CLASSES)


def setup_cfg(args: argparse.Namespace, out_dir: Path):
    from detectron2.config import get_cfg  # type: ignore
    from detectron2.model_zoo import model_zoo  # type: ignore

    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )

    if args.merge_all:
        train_datasets = ("cad_all",)
    elif args.merge_val:
        train_datasets = ("cad_trainval",)
    else:
        train_datasets = ("cad_train",)
    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST  = ("cad_test",)

    cfg.MODEL.WEIGHTS = str(WEIGHTS)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25

    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1024, 1280)
    cfg.INPUT.MAX_SIZE_TRAIN = 2000
    cfg.INPUT.MIN_SIZE_TEST  = 1280
    cfg.INPUT.MAX_SIZE_TEST  = 2000
    cfg.INPUT.RANDOM_FLIP    = "horizontal"

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR       = args.lr
    cfg.SOLVER.MAX_ITER      = args.max_iter
    cfg.SOLVER.STEPS         = (int(args.max_iter * 0.75), int(args.max_iter * 0.90))
    cfg.SOLVER.GAMMA         = 0.1
    cfg.SOLVER.WARMUP_ITERS  = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.AMP.ENABLED   = True

    cfg.TEST.EVAL_PERIOD = 2000

    cfg.OUTPUT_DIR = str(out_dir)
    cfg.freeze()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--eval-only",       action="store_true")
    parser.add_argument("--merge-val",       action="store_true",
                        help="Gộp val vào train để có nhiều data hơn")
    parser.add_argument("--merge-all",       action="store_true",
                        help="Dùng tất cả train+val+test (all.json) để train")
    parser.add_argument("--out-name",        type=str, default="cad_layout",
                        help="Tên thư mục output trong runs/detectron2/")
    parser.add_argument("--max-iter",        type=int,   default=20_000)
    parser.add_argument("--lr",              type=float, default=2.5e-4)
    parser.add_argument("--score-thr",       type=float, default=0.5,
                        help="Score threshold for post-processing")
    parser.add_argument("--containment-thr", type=float, default=0.8,
                        help="Containment threshold for same-class NMS")
    parser.add_argument("--text-merge-thr",  type=float, default=0.2,
                        help="Overlap threshold to merge adjacent text boxes")
    parser.add_argument("--text-padding",    type=float, default=5.0,
                        help="Padding (px) added to each text box after merging")
    args = parser.parse_args()

    out_dir = RUNS_DIR / args.out_name

    from detectron2.data.datasets import register_coco_instances  # type: ignore
    from detectron2.engine import DefaultPredictor, DefaultTrainer  # type: ignore
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset  # type: ignore
    from detectron2.data import build_detection_test_loader  # type: ignore

    sys.path.insert(0, str(Path(__file__).parent))
    from postprocess import apply_postprocess  # type: ignore

    logging.basicConfig(level=logging.INFO)

    for split in ["train", "val", "test"]:
        register_coco_instances(
            f"cad_{split}", {},
            str(COCO_DIR / f"{split}.json"),
            str(IMG_DIR / split),
        )
    # trainval = train + val gộp lại
    register_coco_instances(
        "cad_trainval", {},
        str(COCO_DIR / "trainval.json"),
        str(IMG_DIR / "trainval"),
    )
    # all = train + val + test gộp lại
    register_coco_instances(
        "cad_all", {},
        str(COCO_DIR / "all.json"),
        str(IMG_DIR / "all"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = setup_cfg(args, out_dir)

    print("\n▶  Training config:")
    print(f"   Pretrained  : {WEIGHTS}")
    print(f"   Dataset     : {cfg.DATASETS.TRAIN}")
    print(f"   Classes     : {CLASSES}")
    print(f"   Max iters   : {cfg.SOLVER.MAX_ITER}")
    print(f"   LR          : {cfg.SOLVER.BASE_LR}")
    print(f"   Batch       : {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   Input size  : {cfg.INPUT.MIN_SIZE_TRAIN}")
    print(f"   Output dir  : {cfg.OUTPUT_DIR}\n")

    if args.eval_only:
        cfg = cfg.clone(); cfg.defrost()
        cfg.MODEL.WEIGHTS = str(out_dir / "model_final.pth")
        cfg.DATASETS.TEST  = ("cad_test",)
        cfg.freeze()

        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(
            "cad_test", output_dir=str(out_dir / "eval")
        )
        loader = build_detection_test_loader(cfg, "cad_test")
        results = inference_on_dataset(predictor.model, loader, evaluator)
        print("\n▶  Raw results:", results)

        # ── Post-processing ──────────────────────────────────────────────────
        eval_dir  = out_dir / "eval"
        raw_json  = eval_dir / "coco_instances_results.json"
        post_json = eval_dir / "coco_instances_results_postprocessed.json"

        with open(raw_json) as f:
            raw_preds = json.load(f)
        with open(COCO_DIR / "test.json") as f:
            test_gt = json.load(f)

        cat_name_to_id = {c["name"]: c["id"] for c in test_gt["categories"]}

        # Group by image, apply post-processing, flatten
        from collections import defaultdict as _dd
        by_img: dict = _dd(list)
        for p in raw_preds:
            by_img[p["image_id"]].append(p)

        filtered: list = []
        for img_preds in by_img.values():
            filtered.extend(apply_postprocess(
                img_preds,
                cat_name_to_id=cat_name_to_id,
                score_thr=args.score_thr,
                containment_thr=args.containment_thr,
                text_merge_thr=args.text_merge_thr,
                text_padding=args.text_padding,
            ))

        post_json.write_text(json.dumps(filtered, indent=2))
        print(f"\n▶  Post-processing (score>={args.score_thr}, containment>={args.containment_thr}):")
        print(f"   Raw predictions : {len(raw_preds)}")
        print(f"   After filtering : {len(filtered)}")
        print(f"   Saved to        : {post_json}")
        return

    class CADTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = str(out_dir / "eval")
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    trainer = CADTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
