"""layout_detect.py — Thin wrapper around the existing Detectron2 layout model.

Uses the pre-trained checkpoint at:
  layout_detect/models/checkpoints/cad_layout_v7_swapsplit/model_final.pth

Classes: text | table | title_block | diagram

Returns a list of detected blocks per image.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from cad_pipeline.config import (
    LAYOUT_CLASSES,
    LAYOUT_MAX_SIZE,
    LAYOUT_MIN_SIZE,
    LAYOUT_SCORE_THR,
    LAYOUT_WEIGHTS,
    PROJECT_ROOT,
)

# Add layout_detect scripts to sys.path so grid_engine etc. can be imported
_LAYOUT_SCRIPTS = PROJECT_ROOT / "layout_detect" / "scripts"
if str(_LAYOUT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_LAYOUT_SCRIPTS))


@dataclass
class LayoutBlock:
    """A single detected layout region."""

    label: str          # "text" | "table" | "title_block" | "diagram"
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the block region from a full-page image (BGR numpy array)."""
        return image[self.y1 : self.y2, self.x1 : self.x2]


class LayoutDetector:
    """Singleton-style wrapper around Detectron2 predictor."""

    _instance: "LayoutDetector | None" = None

    def __init__(self, score_thr: float = LAYOUT_SCORE_THR) -> None:
        self._score_thr = score_thr
        self._predictor: Any = None

    @classmethod
    def get(cls, score_thr: float = LAYOUT_SCORE_THR) -> "LayoutDetector":
        if cls._instance is None or cls._instance._score_thr != score_thr:
            cls._instance = cls(score_thr)
        return cls._instance

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.model_zoo import model_zoo
        except ImportError as exc:
            raise ImportError(
                "detectron2 is required for layout detection.\n"
                "Install via: pip install detectron2"
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        )
        cfg.MODEL.WEIGHTS = str(LAYOUT_WEIGHTS)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LAYOUT_CLASSES)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._score_thr
        cfg.INPUT.MIN_SIZE_TEST = LAYOUT_MIN_SIZE
        cfg.INPUT.MAX_SIZE_TEST = LAYOUT_MAX_SIZE
        cfg.freeze()
        self._predictor = DefaultPredictor(cfg)

    def predict_image(self, image: np.ndarray) -> list[LayoutBlock]:
        """Run layout detection on a BGR numpy image.

        Args:
            image: BGR uint8 numpy array (as loaded by cv2.imread).

        Returns:
            List of LayoutBlock sorted top-to-bottom, left-to-right.
        """
        self._load()
        outputs = self._predictor(image)
        instances = outputs["instances"].to("cpu")

        blocks: list[LayoutBlock] = []
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        labels = instances.pred_classes.numpy()

        for box, score, label_id in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            blocks.append(
                LayoutBlock(
                    label=LAYOUT_CLASSES[int(label_id)],
                    score=float(score),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=int(label_id),
                )
            )

        return _sort_blocks(blocks)

    def predict_file(self, image_path: Path | str) -> list[LayoutBlock]:
        """Convenience method: load image from disk, run detection."""
        path = Path(image_path)
        img = cv2.imread(str(path)) if hasattr(cv2, "imread") else None
        if img is None:
            with Image.open(path) as pil_image:
                rgb = pil_image.convert("RGB")
            img = np.asarray(rgb)[:, :, ::-1].copy()
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict_image(img)


def _sort_blocks(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """Sort blocks reading-order: left column before right, then top-to-bottom."""
    if not blocks:
        return blocks
    # Compute page mid-x to split columns
    xs = [(b.x1 + b.x2) / 2 for b in blocks]
    all_x1 = [b.x1 for b in blocks]
    page_width = max(b.x2 for b in blocks)
    # Simple 2-column split at median x
    mid_x = sorted(xs)[len(xs) // 2]
    left = sorted([b for b in blocks if (b.x1 + b.x2) / 2 <= mid_x], key=lambda b: b.y1)
    right = sorted([b for b in blocks if (b.x1 + b.x2) / 2 > mid_x], key=lambda b: b.y1)
    return left + right
