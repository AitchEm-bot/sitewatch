"""Shape-based corroboration of YOLO detections.

- ``confirm_hardhat`` runs HoughCircles on the upper third of a person's bbox
  to check whether a rounded hard-hat silhouette is present. Used to downgrade
  a YOLO "Hardhat" prediction when no circle is found.
- ``detect_safety_cones`` segments orange regions in HSV, applies morphology,
  and returns contour bboxes whose aspect ratio is consistent with a cone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def confirm_hardhat(frame: np.ndarray, person_bbox: tuple[int, int, int, int]) -> bool:
    """Return True if a circular hard-hat silhouette is detected in the upper
    third of the person bbox."""
    if frame is None or frame.size == 0:
        return False
    x1, y1, x2, y2 = person_bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 6:
        return False

    head_h = max(1, (y2 - y1) // 3)
    crop = frame[y1 : y1 + head_h, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    crop_w = crop.shape[1]
    min_r = max(4, int(0.10 * crop_w))
    max_r = max(min_r + 1, int(0.30 * crop_w))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, crop_w // 2),
        param1=80,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    return circles is not None and len(circles) > 0


def detect_safety_cones(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect orange safety cones via HSV thresholding + contour aspect-ratio filter."""
    if frame is None or frame.size == 0:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 150, 150]), np.array([15, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0:
            continue
        # cones are taller than wide
        aspect = h / float(w)
        if 1.1 <= aspect <= 3.5:
            out.append((x, y, x + w, y + h))
    return out


def _cli(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    cones = detect_safety_cones(img)
    print(f"safety cones found: {len(cones)} {cones}")
    overlay = img.copy()
    for x1, y1, x2, y2 in cones:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    out_path = Path(path).with_name(Path(path).stem + "_cones.jpg")
    cv2.imwrite(str(out_path), overlay)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m cv_pipeline.shapes <image_path>", file=sys.stderr)
        sys.exit(1)
    _cli(sys.argv[1])
