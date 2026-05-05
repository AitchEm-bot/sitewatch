"""SiteWatch — Streamlit entry point."""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cv_pipeline.detection import Detector
from cv_pipeline.enhancement import enhance
from cv_pipeline.fusion import LOG_PATH, fuse
from cv_pipeline.motion import MotionDetector
from cv_pipeline.shapes import detect_safety_cones
from utils.drawing import draw_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

st.set_page_config(page_title="SiteWatch", layout="wide")


# ----- cached singletons ----------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO models…")
def get_detector(conf_threshold: float) -> Detector:
    return Detector(conf_threshold=conf_threshold)


# ----- pipeline -------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    detector: Detector,
    motion: MotionDetector,
    *,
    do_enhance: bool,
    do_motion: bool,
) -> tuple[np.ndarray, list, list, list]:
    img = enhance(frame) if do_enhance else frame
    dets = detector.detect(img)
    if do_motion:
        hazards = [d.bbox for d in dets if d.category == "hazard"]
        regions = motion.update(img, hazards)
        persons = [d.bbox for d in dets if d.category == "worker"]
        proximity = motion.check_proximity(persons, regions)
    else:
        regions, proximity = [], []
    anns = fuse(dets, regions, proximity)
    overlay = draw_overlay(img, dets, regions, anns, proximity)
    return overlay, dets, anns, proximity


def _resize_long_edge(frame: np.ndarray, max_long: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= max_long:
        return frame
    scale = max_long / long
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


# ----- sidebar --------------------------------------------------------------
st.sidebar.title("SiteWatch")
do_enhance = st.sidebar.toggle("Enable image enhancement", value=True)
do_motion = st.sidebar.toggle("Enable motion detection", value=True)
conf_threshold = st.sidebar.slider("Detection confidence", 0.2, 0.8, 0.4, 0.05)
proximity_px = st.sidebar.slider("Proximity alert distance (px)", 50, 200, 80, 5)

with st.sidebar.expander("Recent violation log", expanded=False):
    if LOG_PATH.exists():
        rows: list[list[str]] = []
        with LOG_PATH.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                rows.append(row)
        rows = rows[-20:][::-1]
        if rows:
            st.dataframe(
                {h: [r[i] for r in rows] for i, h in enumerate(header or [])},
                use_container_width=True,
                height=300,
            )
        else:
            st.caption("No violations logged yet.")
    else:
        st.caption("No violations logged yet.")


# ----- shared resources ----------------------------------------------------
detector = get_detector(conf_threshold)
detector.conf_threshold = conf_threshold


# ----- tabs ----------------------------------------------------------------
tab_live, tab_review = st.tabs(["Live monitoring", "Review footage"])


with tab_live:
    st.markdown("### Live monitoring")
    st.caption("Webcam feed. Grant camera access when prompted.")

    try:
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        import av  # noqa: F401  (ensures the runtime is available)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "streamlit-webrtc is not available — install requirements.txt to enable live mode."
            f"\n\n({exc})"
        )
    else:

        class SiteWatchProcessor(VideoProcessorBase):
            def __init__(self) -> None:
                self.motion = MotionDetector(proximity_threshold=proximity_px)
                self.last_anns: list = []
                self._times: list[float] = []
                self._frame_idx = 0
                self._skip_alt = False

            def recv(self, frame):
                import av

                self._frame_idx += 1
                # Adaptive frame skipping when FPS is poor.
                if self._skip_alt and self._frame_idx % 2 == 0:
                    return frame

                t0 = time.perf_counter()
                img = frame.to_ndarray(format="bgr24")
                # Standardise live size for performance.
                if img.shape[1] > 640:
                    img = cv2.resize(img, (640, 480))

                overlay, _dets, anns, _prox = process_frame(
                    img,
                    detector,
                    self.motion,
                    do_enhance=do_enhance,
                    do_motion=do_motion,
                )
                self.last_anns = anns

                dt_s = time.perf_counter() - t0
                self._times.append(dt_s)
                if len(self._times) > 30:
                    self._times.pop(0)
                avg = sum(self._times) / len(self._times)
                fps = 1.0 / avg if avg > 0 else 0.0
                self._skip_alt = fps < 10.0
                cv2.putText(
                    overlay,
                    f"FPS {fps:.1f}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                return av.VideoFrame.from_ndarray(overlay, format="bgr24")

        ctx = webrtc_streamer(
            key="sitewatch-live",
            video_processor_factory=SiteWatchProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx and ctx.video_processor and ctx.state.playing:
            st.markdown("**Recent alerts**")
            anns = ctx.video_processor.last_anns or []
            if anns:
                st.write(
                    [{"priority": a.priority, "text": a.text} for a in anns[:5]]
                )
            else:
                st.caption("All clear — no active alerts.")


with tab_review:
    st.markdown("### Review footage")
    upload = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )
    if upload is not None:
        suffix = Path(upload.name).suffix.lower()
        tmp_path = Path("logs") / f"_review_upload{suffix}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(upload.getvalue())

        if suffix in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(tmp_path))
            if img is None:
                st.error("Could not decode image.")
            else:
                img = _resize_long_edge(img)
                motion = MotionDetector(proximity_threshold=proximity_px)
                overlay, dets, anns, _prox = process_frame(
                    img, detector, motion, do_enhance=do_enhance, do_motion=False
                )
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.write(f"{len(dets)} detections, {len(anns)} alerts")
        else:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                st.error("Could not open video.")
            else:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                placeholder = st.empty()
                progress = st.progress(0.0)
                motion = MotionDetector(proximity_threshold=proximity_px)
                violation_count = 0
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    idx += 1
                    frame = _resize_long_edge(frame)
                    overlay, _dets, anns, _prox = process_frame(
                        frame,
                        detector,
                        motion,
                        do_enhance=do_enhance,
                        do_motion=do_motion,
                    )
                    violation_count += sum(1 for a in anns if a.priority <= 2)
                    if idx % 3 == 0:
                        placeholder.image(
                            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                    if total > 0:
                        progress.progress(min(1.0, idx / total))
                cap.release()
                progress.progress(1.0)
                st.success(
                    f"Processed {idx} frames. {violation_count} critical/warning alerts."
                )
