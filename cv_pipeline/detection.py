"""YOLO detection wrapper. Merges COCO baseline with the custom SiteWatch model."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Categories used downstream by fusion / drawing.
CATEGORIES = {
    "ppe_violation": {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"},
    "ppe_compliant": {"Hardhat", "Safety Vest", "Mask"},
    "worker": {"Person", "person"},
    "hazard": {"machinery", "vehicle", "truck", "bus", "car"},
    "marker": {"Safety Cone"},
}


def _category_for(name: str) -> str:
    for cat, members in CATEGORIES.items():
        if name in members:
            return cat
    return "other"


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    source: str  # "coco" or "sitewatch"
    category: str  # ppe_violation | ppe_compliant | worker | hazard | marker | other


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Detector:
    """Runs the COCO YOLO and (when available) the custom SiteWatch YOLO,
    merges the predictions, and deduplicates overlapping boxes."""

    def __init__(
        self,
        coco_path: str = "models/yolov8n.pt",
        custom_path: str = "models/sitewatch_best.pt",
        conf_threshold: float = 0.4,
    ) -> None:
        # Import lazily so the module imports cleanly even before the
        # ultralytics package is installed (useful for unit-test discovery).
        from ultralytics import YOLO

        self.conf_threshold = float(conf_threshold)
        self.coco = YOLO(coco_path)
        self.custom = None
        if Path(custom_path).exists():
            try:
                self.custom = YOLO(custom_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load custom weights %s: %s", custom_path, exc)
        else:
            log.warning(
                "Custom weights %s not found — running with COCO-only detection. "
                "Train via training/train_yolo.ipynb to enable PPE-violation classes.",
                custom_path,
            )

    def _run(self, model, frame: np.ndarray, source: str) -> list[Detection]:
        results = model.predict(frame, conf=self.conf_threshold, verbose=False)
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        names = r.names
        if r.boxes is None:
            return out
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            class_name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            out.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    source=source,
                    category=_category_for(class_name),
                )
            )
        return out

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        merged: list[Detection] = self._run(self.coco, frame, "coco")
        if self.custom is not None:
            custom_dets = self._run(self.custom, frame, "sitewatch")
            # Custom model is trusted over COCO when bboxes overlap heavily;
            # this lets PPE classes (NO-Hardhat etc.) take precedence over a
            # generic COCO "person" box covering the same worker.
            for cd in custom_dets:
                merged = [m for m in merged if _iou(m.bbox, cd.bbox) <= 0.5]
                merged.append(cd)
        return merged


def _cli(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    det = Detector()
    print(f"loaded image {img.shape}")
    out = det.detect(img)
    for d in out:
        print(f"  {d.source:>9}  {d.class_name:<18} {d.confidence:.2f}  {d.bbox}  [{d.category}]")
    print(f"total: {len(out)} detections")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m cv_pipeline.detection <image_path>", file=sys.stderr)
        sys.exit(1)
    _cli(sys.argv[1])
