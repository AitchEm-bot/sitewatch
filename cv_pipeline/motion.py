"""MOG2 background subtraction with binary morphology + proximity-violation logic.

The ``open`` then ``close`` step on the foreground mask satisfies the brief's
"binary morphological operations" capability — see ``MotionDetector.update``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionRegion:
    bbox: tuple[int, int, int, int]
    area: int
    is_active_machinery_zone: bool


@dataclass
class ProximityViolation:
    person_bbox: tuple[int, int, int, int]
    machinery_bbox: tuple[int, int, int, int]
    distance: float
    severity: int  # 1 = critical, 2 = warning


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


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Minimum Euclidean distance between two axis-aligned bboxes (0 if overlapping)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)


class MotionDetector:
    """Persistent MOG2 background subtractor with morphological cleanup."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        min_area: int = 1500,
        proximity_threshold: float = 80.0,
    ) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.min_area = int(min_area)
        self.proximity_threshold = float(proximity_threshold)

    def update(
        self,
        frame: np.ndarray,
        hazard_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[MotionRegion]:
        """Apply MOG2 + binary morphology, return motion regions."""
        if frame is None or frame.size == 0:
            return []

        fg = self._mog2.apply(frame)
        # Binarise (MOG2 with detectShadows=False already returns 0/255, but
        # an explicit threshold guards against the shadow-grey value).
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

        # Brief requirement: binary morphological operations.
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[MotionRegion] = []
        hazard_bboxes = hazard_bboxes or []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bbox = (x, y, x + w, y + h)
            is_machinery_zone = any(_iou(bbox, hb) > 0.3 for hb in hazard_bboxes)
            regions.append(MotionRegion(bbox=bbox, area=area, is_active_machinery_zone=is_machinery_zone))
        return regions

    def check_proximity(
        self,
        person_bboxes: list[tuple[int, int, int, int]],
        motion_regions: list[MotionRegion],
    ) -> list[ProximityViolation]:
        """For each person, find the nearest active machinery zone and emit a
        ``ProximityViolation`` if it is within ``proximity_threshold`` pixels."""
        violations: list[ProximityViolation] = []
        zones = [r for r in motion_regions if r.is_active_machinery_zone]
        if not zones:
            return violations
        for person in person_bboxes:
            nearest = min(zones, key=lambda r: _bbox_distance(person, r.bbox))
            dist = _bbox_distance(person, nearest.bbox)
            if dist >= self.proximity_threshold:
                continue
            severity = 1 if dist <= self.proximity_threshold * 0.4 else 2
            violations.append(
                ProximityViolation(
                    person_bbox=person,
                    machinery_bbox=nearest.bbox,
                    distance=dist,
                    severity=severity,
                )
            )
        return violations
