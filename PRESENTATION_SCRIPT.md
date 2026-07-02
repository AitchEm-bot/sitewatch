# SiteWatch — Presentation Script

> Read top-to-bottom while demoing the app. **[DO]** = action on screen, **[SAY]** = what to say. Timing notes in *italics*. Keep the Streamlit app running (`streamlit run app.py`) before you start.

---

## 0. Opening (30 sec)

**[SAY]** "SiteWatch is a real-time construction-site safety monitor. It watches a camera feed or uploaded footage and flags safety violations — workers missing a hard hat or vest, and workers getting too close to moving machinery — with prioritized, on-screen alerts and a logged audit trail.

Under the hood it's a classic computer-vision pipeline: we enhance each frame, run object detection, add motion and shape analysis, then fuse everything into ranked announcements. Let me show it running, then walk through how each piece was built."

---

## 1. The app / demo (2–3 min)

**[DO]** Show the Streamlit window. Point at the **sidebar** first.

**[SAY]** "Everything is driven from this sidebar. We have four controls:
- **Image enhancement** and **motion detection** — toggles to turn pipeline stages on and off, so you can see each one's contribution live.
- **Detection confidence** — the YOLO threshold; lower catches more but noisier.
- **Proximity alert distance** — how many pixels count as 'too close' to machinery.

And this **Recent violation log** expander reads back the last 20 rows of our CSV audit log — every alert the system has ever raised."

**[DO]** Switch to the **Review footage** tab. Upload `tests/fixtures/demo.mp4` (or `sample.jpg`).

**[SAY]** "There are two modes. **Live monitoring** runs off the webcam in real time. **Review footage** — what I'm showing now — lets you drop in an image or a video and re-process it frame by frame.

Watch the overlay: colored boxes are detections — red for a PPE violation, green for compliant gear, blue for workers, orange for hazards like vehicles. Active machinery zones get a translucent orange fill. When a worker gets too close to a moving hazard, we draw a dashed red line between them, and a critical alert banner appears along the bottom."

**[DO]** Let the progress bar finish. Point at the success message.

**[SAY]** "At the end it reports total frames processed and how many critical or warning alerts fired. Now — how it's actually built."

---

## 2. How we built it — pipeline stage by stage (5–7 min)

**[SAY]** "The whole thing is one function — `process_frame` in `app.py`. It's five stages in sequence: **enhance → detect → motion → fuse → draw**. I'll take them in order."

### Stage 1 — Enhancement (`cv_pipeline/enhancement.py`)

**[SAY]** "First we clean up the frame. Construction sites are harsh — glare, dust, blown-out sun. We convert the image to **LAB color space**, and apply **CLAHE** — Contrast Limited Adaptive Histogram Equalization — to just the **L (lightness) channel**. That locally boosts contrast without wrecking color. We only touch lightness so we don't distort the hues we rely on later for cone detection. Settings are conservative — `clipLimit=2.0`, `tileGridSize=(8,8)`; higher clip limits over-amplify dust and get misread as small objects. There's also an optional bilateral-denoise pass in the function, but it's **disabled in the app** — it adds 30–60 ms/frame at 1280 px for marginal gain. The sidebar 'enhancement' toggle switches the whole CLAHE step on or off."

### Stage 2 — Detection (`cv_pipeline/detection.py`)

**[SAY]** "Then object detection. This is the interesting part: we run **two YOLO models** and merge them.
- A stock **COCO YOLOv8n** (nano) — knows generic classes like *person*, *truck*, *car*, *machinery*.
- A **custom-trained SiteWatch model** — a **YOLOv8m (medium)** fine-tuned on PPE classes: *Hardhat*, *NO-Hardhat*, *Safety Vest*, *NO-Safety Vest*, *Safety Cone*, and more (**25 classes** total). Trained 200 epochs (early-stop patience 30) on a Tesla T4, base weights `yolov8m.pt`.

We map every raw class into a **category** — `worker`, `hazard`, `ppe_violation`, `ppe_compliant`, `marker` — so the rest of the pipeline doesn't care which model or class name produced it.

The merge is the clever bit: when the custom model and COCO both box the same spot, we compute **IoU** — intersection over union — and if they overlap more than 0.5, the **custom model wins**. So a specific *NO-Hardhat* detection overrides a generic COCO *person* box on the same worker. The code also loads gracefully: if the custom weights are missing, it logs a warning and runs COCO-only instead of crashing."

### Stage 3 — Motion + proximity (`cv_pipeline/motion.py`)

**[SAY]** "Next, motion. We use **MOG2 background subtraction** — it learns the static background over ~500 frames and flags what's moving. Raw motion masks are noisy, so we run **binary morphology**: an **open** to erase specks, then a **close** to fill holes. We pull contours, drop anything below a minimum area, and get clean **motion regions**.

A region becomes an **'active machinery zone'** if it overlaps a detected hazard box — so it's not just *something moved*, it's *the excavator is running here*.

Then **proximity**: for each worker, we find the nearest active zone and measure the true edge-to-edge distance between the boxes. Inside the threshold, it's a violation. Very close — under 40% of the threshold — is **severity 1, critical**; otherwise **severity 2, a warning**."

### Stage 4 — Shape corroboration (`cv_pipeline/shapes.py`)

**[SAY]** "We also have classic CV as a sanity check on the neural net.
- `detect_safety_cones` finds cones by **HSV color thresholding** for orange, morphology to clean the mask, then an **aspect-ratio filter** — cones are taller than they are wide.
- `confirm_hardhat` runs **Hough circle detection** on the top third of a worker's box, looking for the rounded silhouette of a helmet — used to downgrade a shaky YOLO hard-hat call.

This shows two independent techniques — deep learning and hand-built geometry — corroborating each other."

### Stage 5 — Fusion (`cv_pipeline/fusion.py`)

**[SAY]** "Finally we fuse every signal into ranked **announcements**, in a strict priority order:
1. Critical — worker dangerously close to machinery
2. Critical — no hard hat
3. Critical — no safety vest
4. Warning — worker entering a machinery zone
5. Info — multiple workers in one active zone

Everything is sorted highest-priority-first, and announcements are appended to `logs/violations.csv` — the audit trail the sidebar reads back. To stop a standing worker from spamming the log every frame, we dedup with a **coarse spatial key** — bucketing box centers into a grid — plus a **4-second time window**: the same violation at the same spot is logged at most once per 4 seconds. The overlay still shows it every frame; only the log is deduped."

---

## 3. How the overlay is drawn — boxes, features & lines (3–4 min)

**[SAY]** "Everything you see on the frame is `draw_overlay` in `utils/drawing.py`. There are really three different kinds of things drawn, and each is computed differently."

### 3a. The detection boxes — coordinates come *from the model*

**[SAY]** "We don't decide where the boxes go — YOLO tells us. Each detection carries an `xyxy` box: the top-left and bottom-right pixel corners, straight out of the network. We just round them to whole pixels."

```python
xyxy = box.xyxy[0].tolist()   # [x1, y1, x2, y2] in pixels, from YOLO
x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
```

**[SAY]** "Then we draw a rectangle at those corners, colored by category — red = PPE violation, green = compliant, blue = worker, orange = hazard — with a filled label bar showing the class name and confidence."

### 3b. The 'feature' boxes — found by color & shape, not a model

**[SAY]** "Some things are found with classic CV instead of the neural net. For all of these, the box is the **bounding rectangle of a contour** — the outline of a blob of pixels.
- **Cones**: threshold for orange in HSV → clean the mask → find contours → keep only blobs taller than wide (aspect ratio 1.1–3.5).
- **Motion regions**: MOG2 flags moving pixels → morphology cleans them → `findContours` traces each moving blob → `boundingRect` wraps it.
- **Hard hats**: crop the top third of a worker box and run Hough circle detection for the round helmet silhouette."

```python
# the shared move: outline of a blob -> its bounding box
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contour)   # -> (x, y, x+w, y+h)
```

### 3c. The proximity lines — endpoints are box *centers*

**[SAY]** "The dashed red line connects a worker to the machinery they're too close to. Its two endpoints are just the **center points** of the two boxes — the average of each box's corners."

```python
px = (person_bbox[0] + person_bbox[2]) // 2   # worker center x
py = (person_bbox[1] + person_bbox[3]) // 2   # worker center y
mx = (machinery_bbox[0] + machinery_bbox[2]) // 2
my = (machinery_bbox[1] + machinery_bbox[3]) // 2
```

**[SAY]** "But a line only appears when there's a *violation*. Back in `motion.py`, for each worker we find the **nearest** active machinery zone, measure the true edge-to-edge gap between the boxes, and only draw if it's under the threshold — critical if it's under 40% of it. So each line always points at the single closest hazard, and only when it's genuinely too close."

**[SAY]** "And the *dashed* look: a normal line is solid, so we walk along it in fixed-length steps and draw only every other segment — the skipped ones leave the gaps. The bottom banner then surfaces the single highest-priority alert."

---

## 4. Live mode & performance (1–2 min)

**[SAY]** "Live mode is the same pipeline behind **streamlit-webrtc**, running per frame off the webcam. Two things make it real-time:
- We **downscale** frames to 640×480 before processing.
- We track a rolling FPS average and, if it drops below 10, kick in **adaptive frame-skipping** — processing every other frame so the feed stays smooth.

FPS is printed in the corner. Measured: **~11.68 FPS live** (beats the 10 FPS bar), and **~5.40 FPS on review** at 1280 px — which is fine because review is offline. There's a `utils/benchmark.py` that measures per-stage latency; the JSON results live in `benchmarks/`."

---

## 5. Wrap (30 sec)

**[SAY]** "So end to end: a five-stage CV pipeline — enhance, dual-model detect, motion-and-proximity, shape corroboration, prioritized fusion — wrapped in a Streamlit app with live and review modes, real-time performance tuning, and a logged audit trail. It's modular: every stage is its own file with its own CLI, and there's a smoke-test suite in `tests/`. Happy to dive into any stage."

---

## Q&A cheat-sheet (quick facts)

| Question | Answer |
|---|---|
| Why two models? | COCO for generic objects (person/vehicle), custom for PPE classes it was never trained on. Neither alone covers the full class set. |
| Isn't two models more compute? | Yes — a second forward pass per frame. The COCO baseline is *nano* (`yolov8n`, fast); the custom PPE model is *medium* (`yolov8m`, ~25.85M params, the heavier one). We downscale to 640×480 + adaptive frame-skip and still hit **11.68 FPS live** (>10 requirement). The one-model alternative needs a merged COCO+PPE dataset — far more training/labeling effort. |
| How do they merge? | IoU > 0.5 → custom model overrides COCO on the same box. |
| Why LAB + CLAHE? | Boost contrast on the lightness channel only, preserving color for cone detection. |
| What makes a "machinery zone"? | A MOG2 motion region that overlaps a detected hazard box (IoU > 0.3). |
| Critical vs warning proximity? | ≤ 40% of threshold distance = critical (sev 1); otherwise warning (sev 2). |
| How is real-time achieved? | 640×480 downscale + rolling-FPS adaptive frame-skip below 10 FPS. |
| Where's the audit trail? | `logs/violations.csv`, surfaced in the sidebar (last 20). |
| No custom weights? | Detector logs a warning and runs COCO-only — no crash. |
| Classic CV role? | Cone HSV detection + Hough-circle hard-hat check — implemented and unit-tested, but see the honesty note below re: wiring. |

---

## ⚠️ Know before you go — what the code *actually* wires up

Say these plainly if asked; don't claim more than the code does.

- **Shape corroboration is not in the live path.** `detect_safety_cones` / `confirm_hardhat` exist, are unit-tested, and run in the benchmark — but `process_frame` never calls them, so cones/hard-hat-circles don't affect the on-screen result today. Honest framing: *"The classical-CV cross-check is built and tested as a module; wiring it into the live fusion step is the next integration step."*
- **Time-based alert dedup is active.** `fuse` combines a spatial `dedup_key` with a `_DEDUP_WINDOW_SEC = 4.0` window: the same violation at the same spot is written to `violations.csv` at most once every 4 seconds, not once per frame. The full ranked list is still *returned* every frame so the on-screen banner stays lit while the violation persists — only the *log* is deduped. Verified: 90 identical-violation frames → 1 log row; distinct locations log separately; the same spot re-logs after the window elapses.
- **Cones aren't announced even when detected.** Even in the benchmark, cone boxes aren't turned into announcements — `marker` category is intentionally not surfaced.

---

## 🎓 CV-examiner questions (deeper — honest answers ready)

*Thinking as a CV examiner, these are the sharp ones. Answers are grounded in what the code does, not what we wish it did.*

**Q: MOG2 assumes a static camera — what happens on a moving/handheld camera or a PTZ?**
A: It breaks — global motion makes the whole frame foreground, so every hazard box becomes an "active zone." SiteWatch implicitly assumes a fixed-mount site camera. For a moving camera you'd switch to ego-motion compensation (homography-warp the previous frame) or a detection-only proximity scheme that doesn't rely on background subtraction.

**Q: You run CLAHE before a network trained on ordinary images — isn't that a train/inference distribution mismatch?**
A: Yes, that's a fair risk. CLAHE changes the input statistics the model saw during training, so it can help in glare/low-contrast but hurt on already-clean frames. That's exactly why it's a **toggle** — we can A/B it live. The principled fix is to apply the same enhancement during training augmentation so train and inference match.

**Q: Your proximity threshold is in pixels. A pixel is a different real-world distance near vs. far from the camera — how is that valid?**
A: It isn't metric — it's a screen-space proxy. 80 px near the lens is a much smaller real gap than 80 px at the back of the scene, so we under-alert far away and over-alert up close. Proper fix: calibrate with a homography to a ground plane (bird's-eye) and measure distance in meters, or normalize by detected person height as a rough scale reference.

**Q: The two-model merge only drops COCO boxes that overlap a custom box (IoU > 0.5). What about two custom detections, or two real objects that genuinely overlap?**
A: Right — there's no cross-model confidence NMS, and we don't dedup custom-vs-custom. Two workers standing close (IoU > 0.5) could have one box suppressed, and each model's internal NMS is independent. A cleaner design pools all detections and runs a single confidence-weighted NMS with class-aware logic.

**Q: 0.5 IoU and 0.3 zone-overlap and 0.4× severity — where did those constants come from?**
A: They're hand-tuned heuristics, not learned. Be honest: they were set by eyeballing the demo footage. A stronger version would tune them on a labeled validation set against precision/recall for the actual safety events.

**Q: How do you know the detector is accurate? Your benchmarks folder is just latency.**
A: Correct — `utils/benchmark.py` measures per-stage *latency*, not accuracy. Detection accuracy comes from `model.val()` in `training/train_yolo.ipynb`: on a **114-image validation set**, aggregate **mAP@0.5 ≈ 0.755** across all 25 classes (mAP@0.5:0.95 ≈ 0.482). Some classes lag — e.g. `Gloves` at mAP@0.5 ≈ 0.381 (small, visually ambiguous). We don't report an end-to-end *safety-event* precision/recall, which would be the right task-level metric.

**Q: No tracking? Detections are per-frame independent?**
A: Yes — there's no tracker (no SORT/ByteTrack, no Kalman). Consequences: no stable per-worker IDs, alerts keyed by a coarse spatial grid so a worker crossing a bucket boundary can re-alert, and no temporal smoothing to reject one-frame false positives. Adding a tracker would give persistent IDs and let us require N-of-M frames before alerting.

**Q: Frame-skipping under load — doesn't that corrupt MOG2's background model?**
A: It feeds MOG2 a discontinuous stream, so its history/adaptation is uneven, and motion between skipped frames is missed. It's a deliberate latency-vs-fidelity trade to hold real-time; a better approach decouples a steady-rate motion thread from the detector.

**Q: HoughCircles for hard hats — hard hats aren't circles from most angles.**
A: Agreed, it's a weak corroborator by design — it's only meant to *suppress* a shaky positive (e.g. a round cap misread as a helmet), not to detect hats from scratch. Params are hardcoded and viewpoint-sensitive; it's a heuristic cross-check, not a classifier. (And per the honesty note, it isn't wired into the live path yet.)

**Q: The cone detector is a fixed HSV range — how robust is that to lighting/white balance?**
A: Not very. The orange range (H 5–15, high S/V) is narrow, so shade, dusk, or auto-white-balance shifts will drop cones or admit false orange (safety vests, rust). CLAHE-on-L helps a bit by stabilizing lightness. Robust versions learn the color model or just let the trained YOLO `Safety Cone` class handle it.

**Q: Why MOG2 over frame differencing, KNN, or optical flow?**
A: MOG2 is a per-pixel Gaussian-mixture model — more robust to gradual lighting change than naive frame differencing, and cheaper than dense optical flow. We set `detectShadows=False` so cast shadows don't register as motion. KNN would be a reasonable swap; optical flow gives motion *direction* but at higher cost, which we didn't need for occupancy zones.

**Q: What dataset was the custom model trained on, and is it class-balanced?**
A: The **Roboflow "Construction Site Safety Image Dataset"** — 25 classes (PPE, workers, vehicles, machinery, cones, etc.), with a 114-image validation split. It's *not* balanced: small/ambiguous classes like `Gloves` (mAP@0.5 ≈ 0.381) and `NO-Mask`/`Ladder` are under-represented, which is why the report flags combining it with the larger **SH17 dataset** (8,099 images, 17 PPE classes) as future work. The model is fine-tuned from `yolov8m.pt` COCO-pretrained weights, so it inherits general object priors.

**Q: Live is 11.68 FPS but review is only 5.40 FPS — why, and is that a problem?**
A: Review runs at 1280-px long edge (vs. live's 640×480) for quality on uploaded footage, so each frame costs more. It's acceptable because review is *offline* — real-time isn't a requirement there; the user gets a progress bar and a summary. The live path, where 10 FPS is required, clears it at 11.68.
