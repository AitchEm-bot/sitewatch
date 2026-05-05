"""Module-level smoke tests for the SiteWatch pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_pipeline.enhancement import enhance
from cv_pipeline.fusion import Announcement, fuse
from cv_pipeline.motion import MotionDetector
from cv_pipeline.shapes import confirm_hardhat, detect_safety_cones

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jpg"


def _load_fixture() -> np.ndarray:
    if FIXTURE.exists():
        img = cv2.imread(str(FIXTURE))
        if img is not None:
            return img
    # Synthetic fallback so the suite is runnable without the fixture image.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (180, 200), (40, 80, 200), -1)  # orange-ish blob
    return img


def test_enhancement_shape_and_dtype():
    img = _load_fixture()
    out = enhance(img, denoise=True)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_shapes_returns_list():
    img = _load_fixture()
    cones = detect_safety_cones(img)
    assert isinstance(cones, list)
    for box in cones:
        assert len(box) == 4


def test_confirm_hardhat_returns_bool():
    img = _load_fixture()
    h, w = img.shape[:2]
    bbox = (10, 10, w - 10, h - 10)
    result = confirm_hardhat(img, bbox)
    assert isinstance(result, (bool, np.bool_))


def test_motion_returns_list_no_error():
    img = _load_fixture()
    md = MotionDetector()
    r1 = md.update(img)
    img2 = img.copy()
    cv2.rectangle(img2, (50, 50), (150, 150), (255, 255, 255), -1)
    r2 = md.update(img2)
    assert isinstance(r1, list)
    assert isinstance(r2, list)


def test_fusion_returns_announcements():
    from cv_pipeline.detection import Detection
    from cv_pipeline.motion import MotionRegion, ProximityViolation

    dets = [
        Detection(
            class_name="NO-Hardhat",
            confidence=0.9,
            bbox=(10, 10, 50, 60),
            source="sitewatch",
            category="ppe_violation",
        ),
        Detection(
            class_name="Hardhat",
            confidence=0.95,
            bbox=(80, 10, 120, 60),
            source="sitewatch",
            category="ppe_compliant",
        ),
    ]
    regions = [MotionRegion(bbox=(200, 200, 280, 280), area=4000, is_active_machinery_zone=True)]
    proxim = [
        ProximityViolation(
            person_bbox=(180, 180, 220, 240),
            machinery_bbox=(200, 200, 280, 280),
            distance=10.0,
            severity=1,
        )
    ]
    anns = fuse(dets, regions, proxim, log=False)
    assert isinstance(anns, list)
    assert all(isinstance(a, Announcement) for a in anns)
    assert all(1 <= a.priority <= 3 for a in anns)
    # ppe_compliant must NOT generate an announcement
    assert all("Hardhat" != a.text for a in anns)


@pytest.mark.skipif(
    not Path("models/yolov8n.pt").exists(),
    reason="COCO weights not yet downloaded; run the app once to fetch them",
)
def test_detection_smoke():
    from cv_pipeline.detection import Detector

    img = _load_fixture()
    det = Detector()
    out = det.detect(img)
    assert isinstance(out, list)
    for d in out:
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert x2 >= x1 and y2 >= y1
