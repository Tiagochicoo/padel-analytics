# Improvement Catalog — Padel Analytics

Comprehensive prioritized catalog of every possible improvement to the 5-model padel
analysis system, organized by category and priority tier.

---

## How to use this document

Each improvement has:
- **Expected impact** — mAP gain, accuracy gain, or speed gain
- **Effort** — S (hours), M (days), L (weeks)
- **Prerequisites** — what must be done first
- **Notes** — risks, trade-offs, alternatives

Improvements are **independent** unless marked as prerequisites. Mix, match, and pick
what makes sense for your accuracy/speed/deadline goals.

---

## Tier 1 — High Impact, Ready Now

These are well-understood improvements with high expected return and low implementation
risk. Start here.

### 1.1 YOLO nano → small upgrade (player detection, body pose)

| Field | Value |
|---|---|
| **Impact** | +3-5% mAP for detection, +2-4% pose mAP for body keypoints |
| **Effort** | M (~3 days training) |
| **Prerequisites** | n-chain training must finish first |
| **Risk** | Low — same hyperparams, only batch drops from 8→4 |
| **Outputs** | `player_s_best.pt`, `bodypose_s_best.pt` |

yolo26s has 3.8× more parameters than yolo26n (10M vs 2.6M). The nano model's
FPN is the bottleneck for small-player accuracy — small adds a wider feature
pyramid that captures fine-grained pose details. Train both new s-models
sequentially via `scripts/queue_s_training.sh` after the n-chain finishes.

**Configs:** `configs/detection_combined_s.yaml`, `configs/bodypose_s.yaml`

---

### 1.2 Court v2 — pseudo-label Plaimaker images

| Field | Value |
|---|---|
| **Impact** | +0.2-0.5% keypoint mAP, better generalization to diverse camera angles |
| **Effort** | M (~4 h auto-label, ~2 h retrain) |
| **Prerequisites** | `court_best.pt` (v1) must exist |
| **Risk** | Low — labels are noisy but 5.5× more data outweighs noise |
| **Outputs** | `court_v2_best.pt` |

3,615 unannotated Plaimaker court images are ready in
`data/datasets/court_plaimaker/`. Run v1 model on them → auto-generate 26-point
labels → merge with existing 788 annotations → retrain. The 5.5× dataset size
increase is the single biggest accuracy lever for the court model.

**Pipeline:**
1. `python scripts/pseudo_label_court.py` — runs court_best.pt on Plaimaker images
2. `python scripts/convert_court_26.py` — merges pseudo-labels with v1 dataset
3. `python src/train.py --task pose_v2` — retrain on ~4,400 images

---

### 1.3 Progressive resizing (body pose)

| Field | Value |
|---|---|
| **Impact** | +1-3% pose mAP, especially for small distant players |
| **Effort** | S (~4 h total) |
| **Prerequisites** | Body pose training must converge at 640 |
| **Risk** | Very low — established technique, no hyperparam changes needed |
| **Outputs** | Fine-tuned bodypose checkpoint at imgsz 960 |

Train the full schedule at imgsz 640, then take the `best.pt` and fine-tune for
10-20 additional epochs at imgsz 960 with batch 2 (4 GB VRAM allows this only
for inference-precise fine-tuning). Players at 960 are ~31×75 px vs ~21×50 px
at 640 — significantly more pixels for the keypoint head.

---

### 1.4 Higher imgsz fine-tune (player detection)

| Field | Value |
|---|---|
| **Impact** | +1-2% mAP for distant/small players |
| **Effort** | S (~4 h) |
| **Prerequisites** | Player detection at 640 must converge |
| **Risk** | Very low |
| **Outputs** | Fine-tuned player checkpoint at imgsz 960 |

Same progressive resizing technique as 1.3. After 640 training completes, fine-tune
last 20 epochs at 960 with batch 2.

---

### 1.5 Class-weight rebalancing (shot classification)

| Field | Value |
|---|---|
| **Impact** | +3-8% mAP for rare classes (Salida de pared, Volea de reves) |
| **Effort** | S (~no extra time; add 2 lines to config) |
| **Prerequisites** | First baseline training of shotclass must finish |
| **Risk** | Low — standard technique, no downside for majority classes |

Current dataset has severe class imbalance (Golpe de derecha: 7,202 annotations;
Salida de pared: 567). Ultralytics supports `class_weights` or per-class loss
reweighting. Add `cls_pw: [1.0, ...]` to config reflecting inverse frequency.
Also add the `DiceLoss` option or focal loss for the classification head.

---

### 1.6 Augmentation tuning (body pose)

| Field | Value |
|---|---|
| **Impact** | +1-2% pose mAP |
| **Effort** | S (config tweak) |
| **Prerequisites** | None |
| **Risk** | Very low |

Current bodypose augmentation is conservative (mosaic=0, mixup=0, scale=0.5).
Try:
- `scale: 0.8` — more aggressive multi-scale training helps small players
- `hsv_s: 0.8`, `hsv_v: 0.5` — stronger color augmentation for lighting variety
- `degrees: 2.0` — slight rotation (±2°) is safe for pose and helps angle variety

---

### 1.7 Label smoothing

| Field | Value |
|---|---|
| **Impact** | +0.5-1% mAP, better calibration |
| **Effort** | S (2 lines in config) |
| **Prerequisites** | None |
| **Risk** | None |

Ultralytics supports `label_smoothing: 0.1`. Reduces overconfidence on training
labels — particularly helpful for noisy Roboflow datasets.

---

## Tier 2 — Medium Impact, Needs Implementation

These require new code, not just config changes. Worth doing after Tier 1.

### 2.1 Shot triple fusion (detection + pose + ball trajectory → MLP)

| Field | Value |
|---|---|
| **Impact** | +5-10% shot classification accuracy |
| **Effort** | L (~1 week) |
| **Prerequisites** | Shotclass model + bodypose model + ball tracker all working |
| **Risk** | Medium — MLP needs labeled data, must be trained separately |

The standalone shotclass model (YOLOv26 detection, 11 classes) misses the
temporal and kinematic context needed to distinguish similar shots (e.g.,
Bandeja vs Remate). Triple fusion: extract features from all 3 sources →
concatenate into a lightweight MLP classifier (2 hidden layers, 256 dim).

**Architecture:**
- Shot detection: bbox + class logits (11-dim)
- Body pose: 17 keypoints × 2 (xy) = 34-dim, normalized per player
- Ball trajectory: last 5 ball positions relative to player (10-dim)
- → MLP (55 → 256 → 256 → 11) with ReLU + dropout 0.3
- Train on padelTracker100 shot labels (~13,250 labeled shot frames)

---

### 2.2 Bounce detection

| Field | Value |
|---|---|
| **Impact** | Enables rules engine (double bounce, in/out) |
| **Effort** | M (~3 days) |
| **Prerequisites** | Ball tracker working, homography working |
| **Risk** | Medium — bounce detection is challenging with ~8px ball |

Detect ball bounces from trajectory:
1. Velocity reversal: ball Y velocity changes sign at the "lowest" point
2. Court intersection: project ball to 2D court via homography → check if
   intersection point is within court bounds
3. Use ball's pixel radius change: ball gets slightly larger near camera
   (perspective) and smallest at far court end → bounce at size minimum

Kalman smoother residuals also spike during bounce → use as detection signal.

---

### 2.3 Online ReID (real-time appearance matching)

| Field | Value |
|---|---|
| **Impact** | Stable player IDs during live feed without offline post-processing |
| **Effort** | L (~1 week) |
| **Prerequisites** | Current offline ReID working |
| **Risk** | Medium — online is fundamentally harder than offline |

Replace/ supplement the offline DINOv2 clustering with real-time appearance
matching. Proposed approach:
1. Extract DINOv2 embeddings for each detection at low cadence (every 5-10 frames)
2. Maintain EMA prototype for each active ID (P1-P4)
3. Assign new detections to closest prototype (thresholded cosine similarity)
4. Warmup period (~30s) assigns temporary IDs, then reclusters to permanent P1-P4
5. Falls back to BoT-SORT IOU matching when appearance is ambiguous

On Jetson, DINOv2 ViT-B/14 runs at ~15ms per crop — acceptable for 4 player
crops every 5 frames (~12ms/frame overhead).

---

### 2.4 Model cadence optimization (Jetson)

| Field | Value |
|---|---|
| **Impact** | +10-20 fps, reduced thermal throttling |
| **Effort** | M (~2 days) |
| **Prerequisites** | All 5 models working on Jetson |
| **Risk** | Low — runs at different rates, not structural change |

Not all models need to run every frame:

| Model | Frame interval | Rationale |
|---|---|---|
| Player detection | Every frame | Tracks fast player movement |
| Court keypoints | Every 60 frames (2s) | Court is static; run slow + smooth |
| Ball tracking | Every frame | Ball is fast (45-60 fps needed) |
| Body pose | Every 3-4 frames | Pose changes slower than player movement |
| Shot class | Event-triggered | Only when ball is near player (racket zone) |

This cadence reduces total model load by ~50%, leaving GPU headroom for
TensorRT INT8 in the future.

---

### 2.5 TTA (Test-Time Augmentation) — flip only

| Field | Value |
|---|---|
| **Impact** | +1-2% mAP at inference |
| **Effort** | S (~2 h) |
| **Prerequisites** | None |
| **Risk** | Low — flip-only TTA is well-known |

Ultralytics supports `augment=True` at predict time with `flipud=0, fliplr=0.5`.
This runs inference twice (original + flipped) and merges predictions. Cost: 2×
inference time. On Jetson, use selectively (every other frame or player crops only).

---

### 2.6 Multi-camera tracking (future)

| Field | Value |
|---|---|
| **Impact** | Full-court coverage, no blind spots |
| **Effort** | XL (weeks) |
| **Prerequisites** | Single-camera system stable |
| **Risk** | High — multi-camera calibration + handoff is complex |

Two overhead cameras covering the full court. Camera calibration + homography
puts all detected objects into a shared 2D court coordinate system. Cross-camera
tracking: objects close in court-space get the same ID. Requires NTP sync
between cameras.

---

## Tier 3 — Lower Priority, Future

These are valuable but depend on Tier 1&2 being solid first.

### 3.1 YOLO nano → medium upgrade (player detection only)

After s is validated, try yolo26m (25.9M params, 9× n). On 4 GB VRAM:
batch 2, imgsz 640. Likely +2-4% over s. Runs on Jetson at ~15ms = 65fps
(within budget). Risk: OOM during training.

### 3.2 Ensemble: n + s fusion

Run nano and small models in parallel, fuse predictions via NMS or weighted
box fusion. Expected +1-2% over s alone. Cost: 1.0 + 1.5 = 2.5× inference time
(small is 1.5× slower than nano). Best for offline analysis, not live.

### 3.3 INT8 quantization (TensorRT)

After FP16 is working, calibrate INT8 engine. Expected: 2× speedup over FP16,
~1-2% accuracy loss. Requires representative calibration dataset (100-500 frames).
Thermal benefit: Jetson runs cooler at INT8, reducing throttling.

### 3.4 DeepStream pipeline

Migrate from the current Python + OpenCV pipeline to NVIDIA DeepStream SDK.
Pros: lower CPU usage, better sustained throughput, hardware decoder (reduce
CPU frame decode), native Jetson support, multi-camera built in. Cons:
significant rewrite (GStreamer plugins, C++/Python bindings).

### 3.5 Knowledge distillation

Use the trained s-model as a teacher for the n-model (or m → s). Student
learns from teacher's soft labels (+ ground truth). Expected: n-model gains
+1-3% — meaning n runs at 1.0× speed with s-level accuracy. Good for Jetson.

### 3.6 Synthetic data generation

Generate synthetic padel court images with Unity/Blender at scale. Vary:
camera angle, lighting, player positions, shot types. This addresses the
single biggest limitation: dataset diversity. Cost: high (Unity developer,
modeling). Benefit: unlimited training data for every model.

### 3.7 Active learning pipeline

During inference on Jetson, upload low-confidence frames for human labeling.
Add to training set → periodic retrain. Ensures model improves over time.
Requires: confidence tracking, upload infrastructure, labeling interface.

### 3.8 Full padel scoring engine

Implement the complete padel scoring rules engine per `docs/scoring_spec.md`:
- Golden Point (no Ad-In/Ad-Out)
- Two-serve rule
- Double bounce rule
- Wall play rules
- Set/Tie-break logic

Requires bounce detection (2.2), serve detection, and rally state machine.

### 3.9 Shot speed estimation

From homography + ball trajectory: compute ball speed at hit/ bounce. Requires
accurate homography (court model), ball tracking (TrackNetV3), and frame timestamps.
Expected error: ±5 km/h.

### 3.10 Heatmap rendering

Accumulate player foot positions → project via homography to 2D court →
Gaussian KDE → overlay on court template. Already partially planned in
`docs/model_training_architecture.md`. Needs dedicated `src/heatmap.py`.

### 3.11 Automated evaluation suite

A script that runs all models on a held-out test set and produces a unified
report: per-model mAP, per-class accuracy, runtime, and a "regression check"
compared to previous model versions. This enables safe iteration without manual
verification.

---

## Summary: Recommended order

| Phase | Improvements | Rough time |
|---|---|---|
| **Phase 1** (now) | Finish n-chain, test on Jetson | ~5 days |
| **Phase 2** (parallel with Jetson test) | s-models for detection + bodypose | ~3 days |
| **Phase 3** | Court v2 pseudo-labeling, shot class rebalancing | ~1 day |
| **Phase 4** | Shot triple fusion, bounce detection | ~2 weeks |
| **Phase 5** | Online ReID, model cadence, TTA | ~1 week |
| **Phase 6** | Scoring engine, heatmaps, eval suite | ~2 weeks |
| **Future** | m-model, ensemble, INT8, DeepStream, synthetic data | On-demand |

Each phase is independent — stop at any point once accuracy meets your bar.
