"""Bounding-box / banner overlay drawing."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from cv_pipeline.detection import Detection
from cv_pipeline.fusion import Announcement
from cv_pipeline.motion import MotionRegion, ProximityViolation


_COLOR = {
    "ppe_violation": (0, 0, 255),
    "ppe_compliant": (0, 255, 0),
    "worker": (255, 0, 0),
    "hazard": (0, 165, 255),
    "marker": (0, 255, 255),
    "other": (200, 200, 200),
}


def _draw_dashed_line(img, p1, p2, color, thickness=2, dash=10) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    steps = int(length // dash)
    for i in range(0, steps, 2):
        t1 = i / steps
        t2 = min((i + 1) / steps, 1.0)
        a = (int(x1 + dx * t1), int(y1 + dy * t1))
        b = (int(x1 + dx * t2), int(y1 + dy * t2))
        cv2.line(img, a, b, color, thickness)


def draw_overlay(
    frame: np.ndarray,
    detections: Iterable[Detection],
    motion_regions: Iterable[MotionRegion],
    announcements: Iterable[Announcement],
    proximity_violations: Iterable[ProximityViolation] = (),
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Active machinery zones — translucent orange fill
    overlay = out.copy()
    for r in motion_regions:
        if r.is_active_machinery_zone:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), thickness=-1)
    cv2.addWeighted(overlay, 0.2, out, 0.8, 0, dst=out)

    # Detection boxes
    for d in detections:
        color = _COLOR.get(d.category, _COLOR["other"])
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 2), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Proximity-violation lines
    for pv in proximity_violations:
        px = (pv.person_bbox[0] + pv.person_bbox[2]) // 2
        py = (pv.person_bbox[1] + pv.person_bbox[3]) // 2
        mx = (pv.machinery_bbox[0] + pv.machinery_bbox[2]) // 2
        my = (pv.machinery_bbox[1] + pv.machinery_bbox[3]) // 2
        _draw_dashed_line(out, (px, py), (mx, my), (0, 0, 255), 2, 12)

    # Bottom banner: latest critical announcement
    crit = next((a for a in announcements if a.priority == 1), None)
    if crit is not None:
        banner_h = 36
        banner = out.copy()
        cv2.rectangle(banner, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(banner, 0.6, out, 0.4, 0, dst=out)
        cv2.putText(
            out,
            crit.text,
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out
