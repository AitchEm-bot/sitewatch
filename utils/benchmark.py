"""Per-stage latency benchmark for the SiteWatch pipeline.

Usage:
    python -m utils.benchmark <video_path> [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

from cv_pipeline.detection import Detector
from cv_pipeline.enhancement import enhance
from cv_pipeline.fusion import fuse
from cv_pipeline.motion import MotionDetector
from cv_pipeline.shapes import detect_safety_cones
from utils.drawing import draw_overlay


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def _resize_long_edge(frame, max_long: int):
    h, w = frame.shape[:2]
    long = max(h, w)
    if max_long <= 0 or long <= max_long:
        return frame
    scale = max_long / long
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def run_benchmark(path: str, limit: int | None = None, max_long: int = 0) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")

    detector = Detector()
    motion = MotionDetector()
    timings: dict[str, list[float]] = {
        "enhancement": [],
        "detection": [],
        "shapes": [],
        "motion": [],
        "fusion": [],
        "drawing": [],
        "total": [],
    }

    n = 0
    t_overall = time.perf_counter()
    while True:
        if limit is not None and n >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        frame = _resize_long_edge(frame, max_long)
        t0 = time.perf_counter()
        enhanced = enhance(frame)
        t1 = time.perf_counter()
        dets = detector.detect(enhanced)
        t2 = time.perf_counter()
        _ = detect_safety_cones(enhanced)
        t3 = time.perf_counter()
        hazards = [d.bbox for d in dets if d.category == "hazard"]
        regions = motion.update(enhanced, hazards)
        persons = [d.bbox for d in dets if d.category == "worker"]
        proximity = motion.check_proximity(persons, regions)
        t4 = time.perf_counter()
        anns = fuse(dets, regions, proximity, log=False)
        t5 = time.perf_counter()
        _ = draw_overlay(enhanced, dets, regions, anns, proximity)
        t6 = time.perf_counter()

        timings["enhancement"].append((t1 - t0) * 1000)
        timings["detection"].append((t2 - t1) * 1000)
        timings["shapes"].append((t3 - t2) * 1000)
        timings["motion"].append((t4 - t3) * 1000)
        timings["fusion"].append((t5 - t4) * 1000)
        timings["drawing"].append((t6 - t5) * 1000)
        timings["total"].append((t6 - t0) * 1000)

    cap.release()
    elapsed = time.perf_counter() - t_overall

    summary = {
        "video": str(Path(path).resolve()),
        "frames": n,
        "max_long": max_long,
        "elapsed_sec": round(elapsed, 3),
        "avg_fps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "stages_ms": {
            stage: {
                "p50": round(_percentile(values, 50), 2),
                "p95": round(_percentile(values, 95), 2),
                "mean": round(statistics.fmean(values), 2) if values else 0.0,
            }
            for stage, values in timings.items()
        },
    }
    return summary


def _print_table(summary: dict) -> None:
    print(f"\nFrames processed: {summary['frames']}")
    print(f"Elapsed: {summary['elapsed_sec']:.3f}s   Avg FPS: {summary['avg_fps']}\n")
    print(f"{'stage':<12} {'p50_ms':>10} {'p95_ms':>10} {'mean_ms':>10}")
    print("-" * 44)
    for stage, t in summary["stages_ms"].items():
        print(f"{stage:<12} {t['p50']:>10.2f} {t['p95']:>10.2f} {t['mean']:>10.2f}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="utils.benchmark")
    p.add_argument("video")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--max-long",
        type=int,
        default=1280,
        help="resize each frame so its longer edge is this many pixels; "
        "matches the review-tab resize in app.py. set to 0 to disable.",
    )
    args = p.parse_args(argv)

    summary = run_benchmark(args.video, args.limit, args.max_long)
    _print_table(summary)

    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
