"""Fuse detection / motion / shape signals into prioritised announcements
and append every announcement to logs/violations.csv."""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cv_pipeline.detection import Detection
from cv_pipeline.motion import MotionRegion, ProximityViolation


LOG_PATH = Path("logs/violations.csv")
_DEDUP_WINDOW_SEC = 4.0


@dataclass
class Announcement:
    priority: int  # 1 = critical, 2 = warning, 3 = info
    text: str
    dedup_key: str
    bbox: tuple[int, int, int, int] | None = None


def _ensure_log_header() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "priority", "text", "dedup_key", "bbox"])


def _append_log(announcement: Announcement) -> None:
    _ensure_log_header()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                dt.datetime.now().isoformat(timespec="seconds"),
                announcement.priority,
                announcement.text,
                announcement.dedup_key,
                ";".join(str(v) for v in announcement.bbox) if announcement.bbox else "",
            ]
        )


def _bbox_key(bbox: tuple[int, int, int, int] | None, bin_size: int = 50) -> str:
    """Coarse spatial bucketing so the same worker doesn't generate a fresh
    dedup_key on every pixel of motion."""
    if bbox is None:
        return "global"
    cx = (bbox[0] + bbox[2]) // 2 // bin_size
    cy = (bbox[1] + bbox[3]) // 2 // bin_size
    return f"{cx}_{cy}"


def fuse(
    detections: Iterable[Detection],
    motion_regions: Iterable[MotionRegion],
    proximity_violations: Iterable[ProximityViolation],
    hat_confirmations: dict[tuple[int, int, int, int], bool] | None = None,
    *,
    log: bool = True,
) -> list[Announcement]:
    """Apply rule-ordered fusion and return ranked announcements (highest
    priority first). Every announcement is appended to the violation log."""
    detections = list(detections)
    motion_regions = list(motion_regions)
    proximity_violations = list(proximity_violations)
    hat_confirmations = hat_confirmations or {}

    out: list[Announcement] = []

    # 1. Critical: severity-1 proximity
    for pv in proximity_violations:
        if pv.severity == 1:
            out.append(
                Announcement(
                    priority=1,
                    text="Worker close to moving machinery",
                    dedup_key=f"prox1_{_bbox_key(pv.person_bbox)}",
                    bbox=pv.person_bbox,
                )
            )

    # 2. Critical: NO-Hardhat
    for d in detections:
        if d.category == "ppe_violation" and d.class_name == "NO-Hardhat":
            out.append(
                Announcement(
                    priority=1,
                    text="Worker without hard hat detected",
                    dedup_key=f"noHat_{_bbox_key(d.bbox)}",
                    bbox=d.bbox,
                )
            )

    # 3. Critical: NO-Safety Vest
    for d in detections:
        if d.category == "ppe_violation" and d.class_name == "NO-Safety Vest":
            out.append(
                Announcement(
                    priority=1,
                    text="Worker without safety vest detected",
                    dedup_key=f"noVest_{_bbox_key(d.bbox)}",
                    bbox=d.bbox,
                )
            )

    # 4. Warning: severity-2 proximity
    for pv in proximity_violations:
        if pv.severity == 2:
            out.append(
                Announcement(
                    priority=2,
                    text="Worker entering machinery zone",
                    dedup_key=f"prox2_{_bbox_key(pv.person_bbox)}",
                    bbox=pv.person_bbox,
                )
            )

    # 5. Info: multiple workers in the same active zone
    workers = [d for d in detections if d.category == "worker"]
    for region in motion_regions:
        if not region.is_active_machinery_zone:
            continue
        x1, y1, x2, y2 = region.bbox
        inside = 0
        for w in workers:
            wx = (w.bbox[0] + w.bbox[2]) // 2
            wy = (w.bbox[1] + w.bbox[3]) // 2
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                inside += 1
        if inside >= 2:
            out.append(
                Announcement(
                    priority=3,
                    text="Multiple workers in active zone",
                    dedup_key=f"multi_{_bbox_key(region.bbox)}",
                    bbox=region.bbox,
                )
            )

    # ppe_compliant detections are intentionally not announced.

    out.sort(key=lambda a: a.priority)
    if log:
        for a in out:
            _append_log(a)
    return out
