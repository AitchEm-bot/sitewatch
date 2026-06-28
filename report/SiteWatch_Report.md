# SiteWatch — Construction Site Safety Monitor

*Real-time computer-vision pipeline for personal protective equipment compliance and worker–machinery proximity alerting*

**Authors:** Hani Moustafa; Ibrahim Abdelkarim; Majid Sharaf; Ali Almaharif; Masleen Castleton

**Subject:** CSCI435 — Computer Vision Algorithms and Systems

**Lecturer:** Dr. Patrick Mukala

**Institution:** University of Wollongong in Dubai

**Date:** May 2026

---

## 1. Introduction

### 1.1 Problem statement

Construction is consistently among the most dangerous sectors in the global labour market. The International Labour Organisation reports that the construction industry accounts for [CITE_ILO_STAT] of all occupational fatalities worldwide, with falls, struck-by-object incidents, and contact with moving machinery dominating the cause-of-death distribution. Personal protective equipment (PPE) compliance — hard hats, high-visibility safety vests, eye protection — is a legal requirement on every active site, yet enforcement is overwhelmingly manual and observational: a supervisor walks the site and corrects violations as they are seen, with obvious limits on how many workers and how much area one person can monitor at once.

Computer vision offers a path to continuous, attention-free supervision. A camera feed processed by a detection model can flag missing PPE the moment it appears in frame, log every incident with a timestamp for end-of-day review, and warn a supervisor to intervene before a near-miss becomes an injury. The technical question is whether such a system can run reliably enough on commodity hardware to be deployed without specialist infrastructure. This project investigates that question by building an end-to-end pipeline that processes either a webcam feed or uploaded footage, identifies PPE violations and proximity hazards, and presents results in a dashboard that a supervisor can use without training.

### 1.2 User story

The intended user is a supervisor at an active construction site. A camera — existing CCTV, a tablet, or the supervisor's laptop webcam — is pointed at a work area. The supervisor opens SiteWatch, selects the *Live monitoring* tab, and grants camera access. The interface displays the feed with bounding boxes overlaid on each detected worker: green when their PPE is compliant, red when a violation is present (no hard hat, no safety vest). When a worker enters a moving-machinery zone, the box is connected to the machinery with a dashed warning line. A banner along the bottom of the frame summarises the highest-priority active alert. Every alert is appended to a CSV log for end-of-day review. The supervisor does not need to watch the screen continuously; the visual cues are designed to draw attention only when something is wrong.

### 1.3 Selected vision capabilities

The system integrates four computer vision capabilities, each chosen to address a specific failure mode of using detection alone.

**Image enhancement.** Construction sites operate outdoors with harsh, uneven illumination — a worker in deep shadow under scaffolding, another in direct mid-afternoon sun, dust scattering the light. CLAHE (Contrast-Limited Adaptive Histogram Equalisation) on the L channel of the LAB colour space normalises local contrast without amplifying chrominance noise, recovering detail in over-exposed and under-exposed regions before the detector sees the frame.

**Object detection.** The core capability — a YOLOv8 model identifies workers, PPE items (hard hats, vests, masks), machinery, and safety cones. Two models are run in parallel: the COCO-pretrained YOLOv8n for general classes (people, vehicles, buses) and a YOLOv8m fine-tuned on a construction-specific dataset for safety-domain classes (`Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `machinery`, `Safety Cone`, etc.). Results are merged with intersection-over-union deduplication so the custom predictions take precedence on overlap.

**Shape detection.** Classical shape priors cross-validate the deep model. HoughCircles applied to the upper third of a worker's bounding box checks for the curved silhouette of a hardhat, suppressing weak YOLO `Hardhat` predictions that lack the expected geometry. HSV thresholding plus contour aspect-ratio filtering identifies orange safety cones independently of the YOLO model. These checks are not the primary detector; they are a sanity layer.

**Motion and binary morphology.** A persistent MOG2 background subtractor flags moving regions of the frame, then `cv2.morphologyEx` with a 5×5 rectangular kernel applies an opening followed by a closing operation to clean isolated noise pixels and bridge fragmented foreground patches. Motion regions that overlap with a YOLO-detected hazard (`machinery`, `vehicle`, `truck`) are marked as active machinery zones. Worker bounding boxes within a configurable distance of these zones produce proximity violations, which are the system's most safety-critical output: a worker close to a moving excavator is at risk in a way no PPE check can capture. The morphological operations explicitly satisfy the brief's requirement for binary morphology — they are not a cosmetic step but are load-bearing in the motion pipeline.

---

## 2. System architecture

### 2.1 Overview

SiteWatch is structured as a thin Streamlit frontend on top of an OpenCV / PyTorch processing pipeline, with each computer vision capability isolated in its own module. The data flow is unidirectional: a frame arrives from either WebRTC (live mode) or a file upload (review mode), passes through enhancement, detection, shape validation, motion analysis, and decision fusion, then exits as an annotated overlay rendered back to the browser. State that needs to persist across frames — the YOLO model weights, the MOG2 background model — is held in cached singletons; per-call state (detections, motion regions) is stateless.

[ARCHITECTURE_DIAGRAM]

### 2.2 Frontend

The frontend uses Streamlit (1.32+) with the `streamlit-webrtc` extension for browser-mediated camera access. Streamlit was chosen over a React-based alternative for two reasons. First, the team's expertise is in Python and computer vision, not JavaScript; building the UI in Python keeps the entire codebase in one language and one virtual environment, which materially reduces the integration surface. Second, Streamlit's `@st.cache_resource` decorator solves the "model reloads on every interaction" problem with a single line of code, where an equivalent React frontend would require an explicit Python backend service exposed over an HTTP or WebSocket API.

The UI is structured as two tabs. *Live monitoring* uses `streamlit-webrtc` to acquire camera frames in the browser and forward them to a `VideoProcessorBase` subclass running in the Streamlit server process; each frame is processed and the annotated output is sent back over WebRTC. *Review footage* accepts a file upload (`.jpg`, `.png`, `.mp4`, `.avi`) and processes it offline, displaying a progress bar and an overall summary on completion. A sidebar exposes the runtime toggles — image-enhancement on/off, motion-detection on/off, detection confidence threshold, proximity-alert distance — and a collapsible expander showing the last 20 rows of the violation log.

### 2.3 Backend pipeline

The pipeline is implemented as five independent modules under `cv_pipeline/`, plus a `utils/` package for drawing and benchmarking. Each module exposes a small, well-typed interface and has no dependency on Streamlit, which means each can be exercised by command-line entry points (every module has a `_cli` function for direct invocation) and by the pytest suite without UI fixtures.

The data flow per frame is deterministic and short:

1. `enhance(frame)` — CLAHE on the L channel of LAB.
2. `detector.detect(enhanced)` — runs both YOLOs, merges with IoU deduplication, returns a list of `Detection` objects.
3. `motion.update(enhanced, hazards)` — applies MOG2 + binary morphology, returns motion regions, marking those overlapping a detected hazard as active machinery zones.
4. `motion.check_proximity(persons, regions)` — returns `ProximityViolation` records for each worker–machinery pair within the threshold distance.
5. `fuse(detections, regions, proximity)` — applies a priority-ordered rule cascade and returns `Announcement` records carrying priority, text, deduplication key, and bounding box.
6. `draw_overlay(...)` — composites bounding boxes, machinery-zone fills, dashed proximity lines, and a critical-alert banner into the output frame.

Stage timings are captured by `utils/benchmark.py` and reported per-stage at p50, p95, and mean. This granularity was essential during optimisation: it identified that motion analysis at native 4K resolution dominated frame time and motivated the introduction of a `--max-long` resize argument that mirrors the resize the live and review tabs already perform.

### 2.4 Models

The detection layer combines two YOLOv8 variants. The smaller `yolov8n.pt` (3.0 M parameters, 8.1 GFLOPs) is the COCO-pretrained baseline distributed by Ultralytics; it covers general-purpose classes — `person`, `truck`, `bus`, `car` — that are present in COCO but absent from the construction safety dataset. The larger `yolov8m.pt` (25.85 M parameters, 78.8 GFLOPs) is fine-tuned on the Roboflow Construction Site Safety dataset and contributes the safety-domain classes (`Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Mask`, `NO-Mask`, `machinery`, `Safety Cone`, and 17 vehicle / equipment subtypes).

Predictions from both models are merged in the `Detector.detect` method. For each custom-model bounding box, any COCO box with intersection-over-union greater than 0.5 against it is removed; the custom box is then appended. This asymmetric merge encodes a deliberate priority — when both models identify the same physical object, the custom model wins, because its safety classes are strictly more informative than COCO's generic `person`. The 0.5 IoU threshold was chosen to be permissive enough to match boxes that overlap on the same worker even if their tightness differs, but strict enough that two adjacent workers are not merged into one.

The choice of `yolov8m` over the smaller `yolov8n` for the custom model was made empirically. An initial fine-tune on `yolov8n` reached mAP@0.5 = 0.543 on the held-out validation set; fine-tuning `yolov8m` under identical training settings improved that to 0.563. The aggregate gain is modest, but the per-class gains are concentrated in classes that drive the application — `Hardhat` (+0.082) and `machinery` (+0.153). The `m` model is approximately 2.5–3× slower at inference than `n`, an acceptable cost given that GPU inference still completes well under 50 ms per frame on the target hardware.

---

## 3. Implementation details

### 3.1 Image enhancement

The enhancement module applies CLAHE to the L (luminance) channel of the LAB representation, leaving chrominance untouched. Isolating the contrast adjustment from colour avoids the saturation shifts that occur when CLAHE is applied directly to RGB.

```python
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance(frame: np.ndarray, denoise: bool = False) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if denoise:
        out = cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)
    return out
```

The `clipLimit=2.0` and `tileGridSize=(8,8)` settings are conservative; aggressive clip limits (4.0+) over-amplify dust and fine texture, which the detector then misreads as small objects. The optional `bilateral` denoising step is exposed in the signature but disabled in the live and review pipelines, because it adds 30–60 ms per frame at 1280-pixel long-edge — significant overhead for a marginal quality gain on this dataset.

### 3.2 Object detection

The `Detector` class encapsulates loading both YOLO models and orchestrating the merge. Class names are mapped to internal categories (`worker`, `ppe_violation`, `ppe_compliant`, `hazard`, `marker`, `other`) so that downstream stages do not need to know the specific label vocabulary.

```python
CATEGORIES = {
    "ppe_violation": {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"},
    "ppe_compliant": {"Hardhat", "Safety Vest", "Mask"},
    "worker": {"Person", "person"},
    "hazard": {"machinery", "vehicle", "truck", "bus", "car"},
    "marker": {"Safety Cone"},
}

def detect(self, frame: np.ndarray) -> list[Detection]:
    if frame is None or frame.size == 0:
        return []
    merged = self._run(self.coco, frame, "coco")
    if self.custom is not None:
        custom_dets = self._run(self.custom, frame, "sitewatch")
        for cd in custom_dets:
            merged = [m for m in merged if _iou(m.bbox, cd.bbox) <= 0.5]
            merged.append(cd)
    return merged
```

The `Detection` data class carries `class_name`, `confidence`, `bbox`, `source` (which model produced it), and `category`. The `source` field is preserved through the pipeline and was rendered in the overlay during debugging — useful for diagnosing whether a missed detection is due to the COCO or custom model failing.

### 3.3 Shape detection

Shape detection runs alongside the deep model and provides classical-CV cross-validation. The `confirm_hardhat` function takes a person bounding box, crops to the upper third (where the head is anatomically expected to be), runs HoughCircles, and returns a boolean. The intent is not to detect hardhats from scratch but to suppress weak YOLO `Hardhat` predictions that lack the expected curved silhouette — for example, a baseball cap detected as a hardhat.

```python
def confirm_hardhat(frame: np.ndarray, person_bbox) -> bool:
    x1, y1, x2, y2 = person_bbox
    head_h = max(1, (y2 - y1) // 3)
    crop = frame[y1 : y1 + head_h, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    crop_w = crop.shape[1]
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(10, crop_w // 2),
        param1=80, param2=20,
        minRadius=max(4, int(0.10 * crop_w)),
        maxRadius=max(int(0.10 * crop_w) + 1, int(0.30 * crop_w)),
    )
    return circles is not None and len(circles) > 0
```

Radii are parameterised as fractions of the crop width rather than fixed pixel values, so the function is invariant to bounding-box scale.

The cone detector is colour-based: HSV `inRange` selects orange pixels, morphology cleans the mask, and a contour aspect-ratio filter (height/width between 1.1 and 3.5) returns boxes consistent with cone geometry. The choice of HSV over RGB matters because shadow on an orange cone changes its RGB triple substantially while leaving the hue narrowly bounded.

### 3.4 Motion detection

Motion detection drives the proximity-violation logic, which is the system's highest-stakes output. A persistent MOG2 background subtractor learns the static background over a 500-frame history and emits a binary foreground mask. The mask is cleaned with an open-then-close pair using a 5×5 rectangular kernel — this is the binary morphological operation required by the brief, and it is functionally critical: without it, the foreground mask is dominated by salt-and-pepper noise and isolated edge pixels that produce hundreds of spurious motion contours per frame.

```python
def update(self, frame, hazard_bboxes=None):
    fg = self._mog2.apply(frame)
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < self.min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bbox = (x, y, x + w, y + h)
        is_machinery_zone = any(_iou(bbox, hb) > 0.3 for hb in hazard_bboxes or [])
        regions.append(MotionRegion(bbox=bbox, area=area,
                                    is_active_machinery_zone=is_machinery_zone))
    return regions
```

The `min_area` filter (default 1500 pixels) discards small motion blobs — a worker's hand waving, a flag in the wind — which the system should not treat as machinery. The IoU-vs-hazard check is what fuses the YOLO and motion outputs: a region is marked as active machinery only if it spatially overlaps with a YOLO `hazard` detection. Pure motion without a corresponding hazard label (e.g. a worker walking in an empty area) does not trigger the proximity logic.

`check_proximity` then computes the minimum Euclidean distance from each worker's bounding box to every active machinery zone. Distances under 40 % of the threshold are flagged as severity 1 (critical); the remaining hits within the threshold are severity 2 (warning). Severities map directly to the priorities used in fusion.

### 3.5 Decision fusion and alert generation

Fusion combines the outputs of detection, motion, and proximity into a ranked list of `Announcement` records. The rules are evaluated in a fixed order, with priorities 1 (critical), 2 (warning), and 3 (informational):

1. Severity-1 proximity violations.
2. NO-Hardhat detections.
3. NO-Safety Vest detections.
4. Severity-2 proximity violations.
5. Two or more workers in the same active machinery zone (informational).

```python
for pv in proximity_violations:
    if pv.severity == 1:
        out.append(Announcement(
            priority=1,
            text="Worker close to moving machinery",
            dedup_key=f"prox1_{_bbox_key(pv.person_bbox)}",
            bbox=pv.person_bbox,
        ))
for d in detections:
    if d.category == "ppe_violation" and d.class_name == "NO-Hardhat":
        out.append(Announcement(
            priority=1,
            text="Worker without hard hat detected",
            dedup_key=f"noHat_{_bbox_key(d.bbox)}",
            bbox=d.bbox,
        ))
```

Each `Announcement` carries a `dedup_key` derived from a coarsely-quantised spatial bucket (50-pixel bins on the bounding-box centre). The bucketing means that the same worker walking around continues to share a dedup_key over short distances, which prevents the same physical violation from being repeatedly logged at different sub-pixel positions. The downstream consumer — the on-screen banner and the `logs/violations.csv` file — uses this key to suppress duplicates within a four-second window.

`ppe_compliant` detections (workers correctly wearing PPE) are intentionally not promoted to announcements, even though they are tracked in the detection output. Surfacing every compliant worker as a banner alert would dominate the supervisor's attention budget; the current design surfaces only deviations from the expected state.

An audio output path was prototyped during development using `pyttsx3` (offline text-to-speech) with a threaded priority queue, dedup window, and rate limit. The module worked but added a non-trivial dependency surface — TTS engines vary by operating system, and the queue introduced cross-thread state — without a clear differentiator over the on-screen banner in the deployment context (a noisy construction site where audio cues are easily lost). It was removed before submission to reduce complexity.

### 3.6 Custom model training

The custom YOLO is fine-tuned on the *Construction Site Safety Image Dataset* from Roboflow Universe (`roboflow-universe-projects/construction-site-safety`), containing 2,801 images annotated across 25 classes — the 10 safety-relevant labels (PPE compliance, NO-* violations, `Person`, `Safety Cone`, `machinery`) and 15 vehicle / equipment subtypes (`Excavator`, `dump truck`, `sedan`, `wheel loader`, etc.). The training / validation split provided by the dataset is preserved (114 validation images, 733 instances).

Training is performed in a Google Colab notebook (`training/train_yolo.ipynb`) on a Tesla T4 GPU. The configuration:

- Base weights: `yolov8m.pt` (COCO-pretrained, 25.85 M parameters).
- Epochs: 200, with early stopping at `patience=30` (training halts when validation mAP stops improving for 30 consecutive epochs).
- Image size: 640 × 640.
- Batch size: 16.
- Optimiser, learning rate, augmentations: Ultralytics defaults (AdamW with auto-tuned learning rate, mosaic and mixup augmentations enabled, `close_mosaic=10`).
- Wall-clock: 60–90 minutes on a T4 with `cache="ram"`.

Final validation metrics, reproduced from the `model.val()` output:

| Metric | Value |
|---|---|
| Precision | 0.755 |
| Recall | 0.482 |
| mAP@0.5 | 0.563 |
| mAP@0.5:0.95 | 0.400 |
| Inference speed (T4, batched) | 8.5 ms / image |

Two design decisions warrant explanation. First, image size was kept at the YOLOv8 default of 640 rather than scaled up to 960 or 1280. The deployed pipeline standardises all inputs to a long edge of 640 (live tab) or 1280 (review tab); training at higher resolution than the deployment target gives marginal real-world gains while doubling training time. Second, `patience=30` is a relatively forgiving early-stopping threshold — it cost some epochs that produced no improvement, but it is the parameter that makes the choice of `epochs=200` safe: in practice, training converged and stopped well before reaching the 200-epoch cap.

### 3.7 Challenges encountered

**Streamlit reruns reloading the model.** Streamlit's execution model re-runs the Python script top-to-bottom on every UI interaction. The naïve implementation reloads the YOLO weights from disk on every slider drag, taking roughly two seconds per interaction. The fix is `@st.cache_resource`, which memoises the model loading: the YOLO weights are loaded once per process and reused across script reruns. The same decorator is used for other expensive singletons. Streamlit's caching mode is process-scoped (not request-scoped), which is exactly the semantics needed here.

**Sub-10-FPS at review-tab resolution.** The brief targets real-time (≥10 FPS) operation. At the review-tab resolution of 1280 long edge, the system measures 5.4 FPS (see Section 4.1). This was investigated and found to be dominated by the OpenCV CPU operations (motion analysis at 40 ms, enhancement at 19 ms) rather than by YOLO inference (67 ms is fast in absolute terms but the quadratic-scaling CPU operations on the larger buffer outweigh it). The mitigation chosen was framing rather than optimisation: the live tab — which is the real-time path that the brief's 10-FPS bar applies to — runs at 640 long edge and meets the requirement (11.7 FPS). Review-tab processing is offline and a sub-10-FPS rate is acceptable for batch analysis.

**Dataset ceiling on the n-to-m model upgrade.** Increasing the model from `yolov8n` (3.0 M parameters) to `yolov8m` (25.85 M, ≈8× compute) yielded only a +0.020 increase in overall mAP@0.5 (0.543 → 0.563). The expected gain — based on published YOLOv8 benchmarks on larger datasets — was +0.08 to +0.12. Investigation showed the bottleneck is the dataset rather than the model: several classes (`mini-van`, `sedan`, `truck`, `trailer`) have fewer than five validation instances, which puts a hard ceiling on per-class mAP regardless of model capacity. The lesson is that for a small, niche-domain dataset, model-size returns diminish quickly. Future improvements should target dataset breadth (additional labels, augmentation, joint training with a larger dataset) rather than further model scaling.

**Audio output prototype.** The audio TTS module (Section 3.5) functioned correctly in development but its operating-system-dependent behaviour and the cross-thread state of its priority queue produced inconsistent results across test machines. The complexity outweighed the user-experience gain over the on-screen banner, particularly given the noisy deployment context. The module was scoped out before submission.

---

## 4. Experiments and results

### 4.1 Quantitative metrics

End-to-end pipeline performance was measured on a 30-second 4K test clip (`tests/fixtures/demo.mp4`, 900 frames at 30 fps) on a Windows 11 laptop with an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.4 PyTorch build. The benchmark harness (`utils/benchmark.py`) records per-stage timings across all 900 frames and reports aggregate statistics. Results at the two resolutions actually used by the deployed application:

| Metric | Live tab (640 long edge) | Review tab (1280 long edge) |
|---|---|---|
| Average FPS | **11.68** | 5.40 |
| Total latency p50 / p95 | 64.5 / 77.6 ms | 131.7 / 221.0 ms |
| Detection p50 / p95 | 48.0 / 60.6 ms | 67.3 / 115.6 ms |
| Motion p50 / p95 | 9.0 / 11.3 ms | 40.0 / 67.0 ms |
| Enhancement p50 / p95 | 4.8 / 5.9 ms | 18.9 / 32.8 ms |
| Drawing p50 / p95 | 1.9 / 2.6 ms | 5.2 / 8.8 ms |

The live-tab figure of 11.68 FPS exceeds the brief's 10 FPS requirement by approximately 17 %. The review-tab figure of 5.40 FPS does not meet that bar, but this resolution is used only for offline processing of uploaded footage where real-time constraints do not apply (a 30-second clip processes in approximately 167 seconds; the user sees a progress bar and a final summary).

Detection latency is the largest single contributor to total frame time at 640 (74 %); at 1280 the relative share drops to 51 %, because OpenCV's CPU-side operations scale roughly with pixel count while GPU-side detection is largely resolution-insensitive (Ultralytics resizes internally to its training size). This is consistent with profiling at native 4K, where detection was 26 ms while motion was 322 ms — the detection's GPU-resident operations are the most predictable component of the system.

Per-class detection accuracy on the 114-image validation set (`yolov8m`, mAP@0.5 reported per class):

| Class | Instances | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|
| Person | 166 | 0.765 | 0.895 | 0.671 |
| Safety Cone | 44 | 0.884 | 0.969 | 0.864 |
| Hardhat | 79 | 0.734 | 0.928 | 0.620 |
| NO-Hardhat | 69 | 0.586 | 0.852 | 0.464 |
| Safety Vest | 41 | 0.692 | 0.915 | 0.537 |
| NO-Safety Vest | 106 | 0.622 | 0.808 | 0.554 |
| Mask | 21 | 0.820 | 1.000 | 0.797 |
| NO-Mask | 74 | 0.507 | 0.707 | 0.459 |
| machinery | 8 | 0.740 | 0.971 | 0.625 |
| Excavator | 12 | 0.689 | 0.692 | 0.667 |
| dump truck | 13 | 0.711 | 0.567 | 0.615 |
| wheel loader | 22 | 0.477 | 0.915 | 0.488 |
| Gloves | 25 | 0.381 | 0.813 | 0.320 |
| Ladder | 10 | 0.525 | 0.556 | 0.628 |
| **Aggregate (all 25 classes)** | **733** | **0.563** | **0.755** | **0.482** |

The PPE-violation classes that drive the application — `NO-Hardhat` (0.586) and `NO-Safety Vest` (0.622) — are accurate enough to be useful as alerts but not so accurate that the system can be relied on as a sole compliance gate. Across the board, precision is higher than recall, meaning when the model fires a violation it is usually correct, but it misses a meaningful fraction of true violations. Operationally this is the right asymmetry: false positives create alert fatigue and erode user trust quickly, false negatives are silent, and a missed violation is recoverable through periodic supervisor review of the CSV log. The headline classes for the proximity logic — `machinery` (0.740) and `Person` (0.765) — are both above 0.7, which is the empirical threshold below which the proximity check produced too many spurious alerts during development testing.

### 4.2 Qualitative results

[SCREENSHOT_PPE_VIOLATION]

[SCREENSHOT_MACHINERY_PROXIMITY]

[SCREENSHOT_SAFETY_CONE_DETECTION]

[SCREENSHOT_GOOD_LIGHTING_VS_ENHANCED]

The four screenshots illustrate the four CV capabilities operating in their intended context: a worker without a hard hat correctly flagged with a red bounding box and a critical-priority banner; a worker entering a translucent-orange machinery zone with a dashed proximity line drawn to the moving equipment; orange safety cones detected by both the YOLO model and the HSV / contour fallback in agreement; and a side-by-side comparison of the same frame before and after CLAHE enhancement, showing the recovery of detail in over-exposed sky and under-exposed shadow.

### 4.3 Failure cases

**Vehicle subclass mAP collapse.** The `mini-van`, `sedan`, and `truck` classes have validation mAP@0.5 of 0.124, 0.111, and 0.599 respectively, with one, 13, and three validation instances each. These numbers are statistically meaningless — a single false positive or negative dominates the per-class metric. The system mitigates this in two ways: first, the fusion logic does not branch on vehicle subtype (a `truck`, `dump truck`, and `wheel loader` are all routed to the `hazard` category for the proximity check); second, the COCO baseline contributes its own `truck`, `bus`, and `car` detections, which are well-trained on hundreds of thousands of images and substantially more reliable than the custom model's vehicle-subtype predictions.

**Hooded worker not classified as NO-Hardhat.** During qualitative testing on the bundled `tests/fixtures/demo.mp4`, a worker wearing an orange hoodie with the hood up was not classified as `NO-Hardhat` by either model. The dataset's `NO-Hardhat` examples predominantly show workers with bare heads (visible hair, open foreheads); a hood occluding the head is out-of-distribution and the model defaults to no detection rather than firing. This is a class-boundary problem rather than a model-capacity problem; expanding the training set with hooded examples would close it. As a partial mitigation, the worker is still labelled as `Person` (the COCO model is unaffected), so the supervisor retains visual context to make the call.

**Sub-10-FPS on the review tab at 1280-pixel input.** Discussed in Section 3.7. The chosen mitigation is the resolution split between live and review tabs. The offline nature of review-tab processing makes real-time latency a non-functional requirement in that path, and the live tab — where the requirement does apply — meets it.

---

## 5. Individual contributions

| Name | Tasks | Approximate % |
|---|---|---|
| Hani Moustafa | Decision fusion and alert prioritisation; Streamlit app integration; project coordination | 20% |
| Ibrahim Abdelkarim | Object detection module (COCO + custom YOLO) and model training notebook | 20% |
| Majid Sharaf | Image enhancement (CLAHE / denoise) and shape detection (Hough) | 20% |
| Ali Almaharif | Motion and change detection (MOG2 + morphology + proximity alerting) | 20% |
| Masleen Castleton | Benchmarking, test suite, and report writing | 20% |

---

## 6. Conclusion and future work

### 6.1 Summary

SiteWatch combines four computer vision capabilities — image enhancement (CLAHE on LAB), object detection (dual YOLOv8n COCO + YOLOv8m fine-tuned), shape detection (HoughCircles + HSV / contour), and motion analysis (MOG2 with binary morphological cleanup) — into a single Streamlit application that monitors construction-site footage for PPE compliance and worker–machinery proximity. The custom YOLOv8m achieves mAP@0.5 = 0.563 on a 733-instance validation set, with the application-critical classes — `Hardhat` (0.734), `NO-Hardhat` (0.586), `Safety Vest` (0.692), `NO-Safety Vest` (0.622), `Safety Cone` (0.884), `machinery` (0.740) — all in the usable range. End-to-end live-mode performance on a laptop-class GPU (RTX 4060) is 11.7 FPS, exceeding the brief's 10 FPS bar.

The system is reproducible: the trained weights are bundled in the repository, the demo video and a representative single frame are committed as test fixtures, the training notebook runs end-to-end on free Colab compute, and the benchmark harness produces JSON output committed alongside the README's quoted numbers. A reviewer can clone the repository and reach a working demonstration in under five minutes.

### 6.2 Future work

The most consequential direction — and the one the project is genuinely positioned to enable — is a temporal predictive risk model fed by multi-camera input. The current system reacts to safety events that have already occurred; it flags a missing hard hat the moment it appears in frame, but it cannot predict that a worker is on a trajectory toward a machinery zone five seconds before they arrive. Aggregating detections and proximity violations across multiple camera views and across time would enable a per-zone risk score that updates continuously, allowing supervisors to intervene before a near-miss. The data is already being captured (the violations log is the seed corpus); the missing components are temporal aggregation, camera registration, and a learned mapping from the aggregated state to a forward-looking risk estimate. This is the natural successor system, and the present system is the infrastructure it would be built on.

Other directions, ranked by effort-to-value:

- **Edge deployment.** The pipeline is small enough to run on a Jetson Nano or comparable edge device. Replacing the Streamlit frontend with a headless Python service that streams overlays to a local display would remove laptop tethering and reduce per-camera cost by an order of magnitude.
- **Site access integration.** A turnstile camera that gates entry on PPE compliance would make the system preventative rather than reactive at the perimeter — workers cannot enter without their PPE on.
- **Expanded PPE classes.** Eye protection, harness compliance for elevated work, and improved glove detection (the current `Gloves` class achieves only mAP@0.5 = 0.381 owing to small object size and visual ambiguity). Each new class needs both labelled data and a reliable on-screen indicator.
- **Joint training with a larger dataset.** The dataset ceiling discussed in Section 3.7 is the principal bottleneck on the existing classes. The SH17 dataset (8,099 images, 17 PPE classes) is approximately three times the size of the current Roboflow set and could be combined via either joint training or sequential fine-tuning to improve the underrepresented classes (`Gloves`, `NO-Mask`, `Ladder`).

---

## 7. References

[1] G. Jocher, A. Chaurasia, and J. Qiu, *Ultralytics YOLOv8*, version 8.4.46, 2024. [Online]. Available: https://github.com/ultralytics/ultralytics

[2] Roboflow Universe Projects, "Construction Site Safety Image Dataset," *Roboflow Universe*, 2023. [Online]. Available: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

[3] G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, 2000.

[4] Streamlit Inc., *Streamlit: A faster way to build and share data apps*. [Online]. Available: https://streamlit.io

[5] Y. Yamamoto, *streamlit-webrtc: Real-time video and audio processing on Streamlit*. [Online]. Available: https://github.com/whitphx/streamlit-webrtc

[6] Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," in *Proc. IEEE 17th International Conference on Pattern Recognition (ICPR)*, vol. 2, pp. 28–31, 2004.

[7] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, P. Heckbert, Ed., Academic Press, 1994, pp. 474–485.

[8] N. M. Bhat, *pyttsx3: Offline text-to-speech for Python*. [Online]. Available: https://github.com/nateshmbhat/pyttsx3

[9] [CITE_ILO] International Labour Organization, *Construction: a hazardous work*. [Online]. Available: https://www.ilo.org/

[10] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, real-time object detection," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 779–788, 2016.

[11] T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in *Proc. European Conference on Computer Vision (ECCV)*, pp. 740–755, 2014.
