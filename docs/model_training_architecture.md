# Model Training Architecture

Training pipeline for a production-grade padel game analysis system (Clutchapp.io-class).
Five YOLOv26 models trained sequentially on a single RTX 3050 Ti 4GB, deployed on
Jetson Orin Nano via TensorRT. One-week training window before live camera test.

---

## Infrastructure

| Component | Detail |
|---|---|
| **Train GPU** | RTX 3050 Ti Laptop (4 GB VRAM), CUDA 12.4, driver 595.71.05 |
| **Python** | 3.12.13 venv via uv |
| **Framework** | Ultralytics 8.4.79 + PyTorch 2.6.0+cu124 |
| **Deploy target** | Jetson Orin Nano (TensorRT FP16) |
| **Training entry** | `python src/train.py --task <name>` (reads `configs/<name>.yaml`) |
| **Queue system** | `scripts/queue_all_training.sh` – sequential, GPU never idle |
| **Monitoring** | Web dashboard at `/training` (FastAPI + Chart.js, auto-reloads every 5 s) |

### Training dataset summary

| Model | Task | Train | Val | Total | Source |
|---|---|---|---|---|---|
| Detection | YOLOv26 (detect) | 28,242 | 8,678 | 36,920 | Roboflow octovar + padelTracker100 |
| Court keypoints (26-pt) | YOLOv26-pose | 596 | 192 | 788 | Roboflow joshs-workspace (26 kpts: cage+court+net+service) |
| Ball detection (TrackNetV3) | TrackNetV3 (heatmap) | 224 rallies | 40 rallies | 264 | padelTracker100 ball JSONs → rally chunks |
| Body pose | YOLOv26-pose | 84,847 | 14,975 | 99,822 | padelTracker100 (2 match videos, 49 GB frames) |
| Shot classification | YOLOv26 (detect) | 6,986 | 1,973 | 8,959 | Roboflow padel-ball-hit v1 |

### Additional data (for court model v2 via pseudo-labeling)

| Dataset | Images | Purpose |
|---|---|---|
| Plaimaker/padel (Roboflow) | 3,615 | Unannotated court images → auto-label with v1 model → retrain v2 |
| ghalichraibi (Roboflow) | 142 | Skipped (10-kpt schema, incompatible with 26-pt) |

---

## Data Preparation Pipeline

### Dataset acquisition

```
Roboflow API ──┬──> data/datasets/shotclass_roboflow/    (yolov8 format, 11 classes)
               ├──> data/datasets/ball_roboflow/          (yolov8 format, Tennis Ball)
               ├──> data/datasets/players/                (octovar/player-padel-dataset)
               ├──> data/datasets/court_26_josh/          (joshs-workspace, 26 kpt, 793 imgs)
               └──> data/datasets/court_plaimaker/        (3,615 unannotated court images)

padelTracker100 (Zenodo) ──> data/datasets/padeltracker100/raw/
    ├── 2022_BCN_FinalF_1.mp4 (2.9 GB, 45,934 frames, 30 fps)
    ├── 2022_BCN_FinalM_1.mp4 (4.3 GB, 53,953 frames, 30 fps)
    ├── labels/*_pose.json   (COCO 17-keypoint, 398,917 person instances)
    ├── labels/*_ball.json   (COCO ball, 57,291 annotations, ~8×8 px)
    └── labels/*_shots.csv   (per-frame shot labels, 13,250 shot frames)
```

### Frame extraction (body pose)

```bash
python scripts/extract_frames.py
# 8 workers, ~320 fps, 99,887 frames, 49 GB JPG (1920×1080)
```

### Body pose: COCO → YOLO-pose conversion

```bash
python scripts/convert_coco_pose_to_yolo.py
# 84,847 train / 14,975 val, symlinks to extracted frames
# Temporal split (85/15 per video) to avoid frame leakage
```

### Ball dataset: Roboflow + padelTracker100 merge

```bash
python scripts/build_ball_dataset.py --stride 5
# Subsamples padelTracker100 ball every 5th frame (consecutive frames nearly identical)
# Remaps all Roboflow classes to single "ball" (class 0)
# 13,196 train / 1,233 val
```

### Player detection dataset (downloads + merge)

```python
# scripts/build_combined_dataset.py (run once before training)
# Downloads Roboflow octovar + converts padelTracker100 YOLO files
# Concatenates into data/datasets/combined/
# 28,242 train / 8,678 val
```

### Court keypoints (26-point): conversion

```bash
python scripts/convert_court_26.py
# Converts joshs-workspace 26-class/1-keypoint format → standard 1-class/26-keypoint
# Each image: 26 landmark detections merged into single court instance
# Missing landmarks filled with (0,0,0) — not labeled
# 596 train / 192 val, output → data/datasets/court_keypoints_26/
```

---

## Model 1 — Player Detection (running, epoch 81/100)

Detects padel players (single class) with bounding boxes. Feeds into BoT-SORT
tracking → CourtFilter (inside-court rejection) → PlayerRegistry (4-slot assignment)
→ ReID (global K=4 mapping).

**Config:** `configs/detection_combined.yaml`

| Parameter | Value |
|---|---|
| Base weights | `yolo26n.pt` |
| imgsz | 640 |
| Batch | 8 |
| Epochs | 100 |
| Optimizer | auto (SGD w/ momentum) |
| LR schedule | cosine decay, warmup 3 epochs |
| Patience | 30 |
| Amp | true |
| **Augmentation** | |
| Mosaic | 1.0, close at epoch 15 |
| Mixup | 0.15 |
| Copy-paste | 0.15 |
| HSV | h=0.015, s=0.7, v=0.4 |
| Scale | 0.5 |
| Flip LR | 0.5 |
| Translate | 0.1 |

**Status:** Training, epoch 81/100. ETA ~3.5 h remaining.
**Output:** `data/models/player_best.pt`
**Metric targets:** mAP50-95 > 0.80, Precision > 0.90, Recall > 0.90

---

## Model 2 — Court Keypoints, 26-Point (queued after detection)

Detects 26 court landmarks (cage corners, court corners, net, service line) →
computes overdetermined homography (26 reference points) → projects player/ball
positions to a 20×10 m top-down court map. Used for inside-court player filtering,
2D minimap, serve-box detection, and heatmap projection.

**Config:** `configs/pose.yaml`

| Parameter | Value |
|---|---|
| Base weights | `yolo26n-pose.pt` |
| imgsz | 640 |
| Batch | 8 |
| Epochs | 600 |
| Optimizer | AdamW, lr0=0.001 |
| LR schedule | cosine decay, warmup 3 epochs |
| Patience | 80 |
| kpt_shape | [26, 3] |
| flip_idx | [2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13, 17,16, 19,18, 20,21, 24,25,22,23] |
| Loss weights | pose=12.0, kobj=1.0, box=7.5, cls=0.3, dfl=1.5 |
| **Augmentation** | |
| Mosaic | 0.0 (distorts court polygon) |
| Flip UD / LR | 0.0 / 0.0 |
| Scale | 0.3 |
| Translate | 0.05 |

**26 keypoint indices:**

| Range | Landmarks | Count |
|---|---|---|
| 0–7 | Cage corners (top/bottom × left/right × close/far) | 8 |
| 8–15 | Court corners (top/bottom × left/right × close/far) | 8 |
| 16–19 | Net corners (top/bottom × left/right) | 4 |
| 20–25 | Service line (centre/left/right × close/far) | 6 |

**Why 26 instead of 10:**

- Previous 10-point schema had unnamed indices — impossible to merge with other datasets without error-prone visual matching
- 26 named landmarks give zero ambiguity → safe to combine datasets
- Overdetermined homography (26 points vs minimum 4) = more robust projection
- Richer court polygon: cage + court + net + service line → better inside-court filtering

**Data notes:**

- Source: joshs-workspace-p1aa0/padel-court-detection (793 images, converted from 26-class/1-keypoint to 1-class/26-keypoint format via `scripts/convert_court_26.py`)
- 596 train / 192 val after conversion
- `src/utils/homography.py` defines the 26-point reference template on a 20×10 m court

**Pseudo-labeling plan (v2):**
After v1 trains, run it on 3,615 Plaimaker court images → auto-generate 26-point labels → retrain on combined ~4,400 images for higher accuracy and more diverse camera angles.

**Status:** Queued via `scripts/queue_court_training.sh`.
**Duration:** ~1.5 h (early stopping likely at ~150–250 epochs).
**Output:** `data/models/court_best.pt`
**Metric target:** Keypoint mAP > 0.90

---

## Model 3 — Ball Detection, TrackNetV3 (queued after YOLO chain)

Detects the ball via **heatmap regression** (not bounding box detection). The ball
is ~8×8 px at 1920×1080 and moves at high speed — too small and fast for YOLO's
bounding-box approach. TrackNetV3 uses a 2D-UNet over 8 stacked RGB frames + a
median background frame to predict a Gaussian heatmap, giving sub-pixel ball
position. Purpose-built for small fast balls (tennis, badminton, shuttlecock).

**Config:** `configs/ball.yaml` · **Training:** `src/train_tracknet.py` (NOT `src/train.py`)

| Parameter | Value |
|---|---|
| Architecture | TrackNetV3 (2D-UNet heatmap regressor) |
| Input | 8 stacked RGB frames + 1 median-bg = 27 channels (288×512) |
| Output | 8 heatmaps (Gaussian at ball center, sigma=2.5px) |
| Pretrained | Shuttlecock TrackNetV3 weights (MIT, fine-tuned on padel) |
| Epochs | 30 |
| Batch | 8 |
| Optimizer | Adam, lr=0.001 |
| Loss | WBCELoss (weighted binary cross-entropy) |
| Mixup | alpha=0.5 (Beta distribution) |
| seq_len | 8 frames per input window |
| bg_mode | concat (median background prepended) |

**Why TrackNetV3, not YOLOv26:**
- Ball at 8px → 2.7px at imgsz 640 — below YOLO's reliable detection threshold
- TrackNetV3's heatmap regression handles sub-5px objects via Gaussian modeling
- 8-frame temporal context captures motion blur and trajectory (critical for fast padel balls)
- Sub-pixel output precision → better bounce detection and trajectory analysis
- Proven in tennis/badminton/soccer analytics (de facto standard for ball tracking)

**Data:**
- padelTracker100 ball JSONs → converted to rally chunks (256 frames each)
- 224 train rallies / 40 val rallies, each with consecutive frame triplets + Gaussian heatmap labels
- Built by `scripts/build_tracknet_dataset.py`
- Pretrained on shuttlecock data → fine-tuned on padel

**Status:** Queued via `scripts/queue_ball_tracknet.sh` (waits for YOLO chain to finish).
**Duration:** ~8–12 h.
**Output:** `data/models/ball_best.pt` (promoted from `runs/tracknet/TrackNet_best.pt`)
**Inference:** `src/ball_tracker.py` — TrackNetBallTracker, wired into PadelAnalyzer

---

## Model 4 — Body Pose (running now, epoch 1/100)

Detects 17 COCO body keypoints per player (nose, eyes, shoulders, elbows, wrists,
hips, knees, ankles). Provides per-player skeleton data for shot classification,
swing detection, and biomechanical analysis.

**Config:** `configs/bodypose.yaml`

| Parameter | Value |
|---|---|
| Base weights | `yolo26n-pose.pt` |
| imgsz | 640 |
| Batch | 8 |
| Epochs | 100 |
| Optimizer | auto |
| LR schedule | cosine decay, warmup 3 epochs |
| Patience | 30 |
| kpt_shape | [17, 3] |
| flip_idx | [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15] |
| Loss weights | pose=12.0, kobj=1.0, box=7.5, cls=0.3, dfl=1.5 |
| **Augmentation** | |
| Mosaic | 0.0 (distorts small player keypoints) |
| Scale | 0.5 (multi-scale helps with ~63 px players) |
| Flip LR | 0.5 (flip_idx handles left/right swap) |

**Data notes:**

- Largest training dataset: 99,822 images from 2 full match videos.
- At 30 fps, consecutive frames are nearly identical — validated that full density
  is retained for maximum accuracy (no subsampling). This extends training to ~58 h.
- Players are ~63×149 px at 1920×1080 (≈21×50 px at imgsz 640). The nano model's
  FPN handles this scale, but accuracy on small keypoints may benefit from higher
  imgsz in a future iteration.

**Status:** Queued via `scripts/queue_all_training.sh`.
**Duration:** ~58 h.
**Output:** `data/models/bodypose_best.pt`
**Metric targets:** Pose mAP > 0.85, keypoint visibility accuracy > 0.90

---

## Model 5 — Shot Classification (queued after body pose)

Detects player shots and classifies the type (11 classes). Runs standalone at
inference; also fuses with body pose keypoint features for higher-accuracy
shot type prediction.

**Config:** `configs/shotclass.yaml`

| Parameter | Value |
|---|---|
| Base weights | `yolo26n.pt` |
| imgsz | 640 |
| Batch | 8 |
| Epochs | 200 |
| Optimizer | auto |
| LR schedule | cosine decay, warmup 3 epochs |
| Patience | 50 |
| **Augmentation** | |
| Mosaic | 1.0, close at epoch 15 |
| Mixup | 0.15 |
| HSV | h=0.015, s=0.7, v=0.4 |
| Scale | 0.5 |
| Flip LR | 0.5 |

**Classes (11):**

| ID | Name | English | Annotations |
|---|---|---|---|
| 0 | Bandeja | Defensive overhead (≈víbora) | 1,951 |
| 1 | Bola | Ball (non-shot) | – |
| 2 | Contrapared | Wall rebound after shot | 1,192 |
| 3 | Globo | Lob (high defensive arc) | 2,308 |
| 4 | Golpe de derecha | Forehand | 7,202 |
| 5 | Golpe de reves | Backhand | 6,302 |
| 6 | Remate | Smash | 3,011 |
| 7 | Salida de pared | Wall exit shot | 567 |
| 8 | Saque | Serve | 3,236 |
| 9 | Volea de derecha | Forehand volley | 1,573 |
| 10 | Volea de reves | Backhand volley | 1,184 |

(Approximate per-class annotation counts from the 9,900-image dataset.)

**Status:** Queued via `scripts/queue_all_training.sh` (runs after body pose).
**Duration:** ~10 h.
**Output:** `data/models/shotclass_best.pt`
**Metric target:** mAP50 > 0.90 per class, balanced accuracy across shot types

---

## ReID — Stable Player Identities

Does NOT require training. Runs offline after video analysis.

| Component | Detail |
|---|---|
| Backbone | DINOv2 ViT-B/14 (768-dim) via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` |
| Fallback | OSNet (if torchreid installed), else ResNet50 (torchvision) |
| Algorithm | Constrained agglomerative clustering (K=4) |
| Constraints | Temporal cannot-link (same tracklet can't be two IDs) + position prior (players stay in court quadrant) |
| Config | `configs/botsort_reid.yaml` (shared with live BoT-SORT tracking — same conf/iou/imgsz) |
| Cache | `data/reid_cache/<source>_<size>_<weights>_<tracker>_k4.json` |
| Server | Background compute via `/api/reid`, status in `/stats` |
| Integration | Auto-loaded by `build_analyzer` → `PadelAnalyzer.apply_reid_mapping()` |

**Key design:** Online tracking uses the same BoT-SORT config as the ReID feature
extraction, ensuring raw tracklet IDs are consistent between live and offline passes.

---

## Training Queue & Timeline

All models run sequentially to saturate the GPU. Detection is running; the rest
are queued via `scripts/queue_all_training.sh` and `scripts/queue_court_training.sh`.

```
Day 1 (Fri)     Day 2 (Sat)    Day 3 (Sun)     Day 4 (Mon)     Day 5-7 (Tue–Thu)
┌──────────────┐
│ Detection    │  ✅ DONE (100 epochs, player_best.pt)
│ 36,920 imgs  │
└──┬───────────┘
   │ court pose 26pt ── ✅ DONE (117 epochs, mAP 0.995, court_best.pt)
   │  788 imgs
   └─────────────────┐
                     │ body pose ──────────────────── RUNNING NOW (~120 h)
                     │ 99,822 imgs, epoch 1/100
                     └─────────────────────────────────────┐
                                                                                      │ shot class ─ (~10 h)
                                                                                      │ 8,959 imgs, 11 classes
                                                                                      └──────────┐
                                                                                                │
                                                                                      TrackNetV3 ball (~8-12 h)
                                                                                      264 rally chunks, heatmap regression
                                                                                                │
                                                                                     EVAL + ITERATE
                                                                                     Pseudo-label Plaimaker → court v2
                                                                                     TensorRT export
                                                                                     Live camera test
                                                    ↑
                                               Monday evening:
                                               All 5 models finished
```

### Total compute: ~101 h continuous (4.2 days)

After training (days 5–7):

- Pseudo-label 3,615 Plaimaker images with court v1 → retrain court v2 (~4,400 imgs)
- Evaluate all 5 models on held-out test data
- Retune hyperparameters for underperformers (scale/lr0/augmentation)
- Export to TensorRT FP16 → deploy on Jetson Orin Nano

---

## Inference Pipeline (Jetson Orin Nano)

All 5 models run per frame, then fanned into the rules engine.

```
                    ┌─────────────────────┐
  frame ───────────▶│  YOLOv26 detection  │──▶ 4 player bboxes
                    └─────────┬───────────┘        │
                              │                    │
                    ┌─────────▼───────────┐        │
                    │  YOLOv26-pose       │        │
                    │  (court keypoints)  │──▶ homography → 2D map
                    └─────────┬───────────┘        │
                              │                    │
                    ┌─────────▼───────────┐        │
                    │  YOLOv26            │        │
                    │  (ball detection)   │──▶ ball position
                    └─────────┬───────────┘        │
                              │                    │
                    ┌─────────▼───────────┐        │
                    │  YOLOv26-pose       │        │
                    │  (body keypoints)   │──▶ 17 kpts × 4 players
                    └─────────┬───────────┘        │
                              │                    │
                    ┌─────────▼───────────┐        │
                    │  YOLOv26            │        │
                    │  (shot class)       │──▶ shot type (11 classes)
                    └─────────┬───────────┘        │
                              │                    │
                              ▼                    ▼
                    ┌─────────────────────────────────────┐
                    │         Rules Engine                │
                    │  • Ball bounce detection             │
                    │  • Rally state machine               │
                    │  • Scoring per padel rules           │
                    │  • Serve detection                   │
                    │  • Shot speed from homography        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │            Stats Output              │
                    │  • Heatmaps (2D Gaussian KDE per ID) │
                    │  • Shot counts by type per player    │
                    │  • Rally length + outcome            │
                    │  • Point score                       │
                    │  • Average shot speed                │
                    │  • Player movement maps              │
                    └─────────────────────────────────────┘
```

### TensorRT export

```bash
# On Jetson (after scp'ing .pt files from laptop)
python src/export_trt.py \
    --weights data/models/player_best.pt \
    --imgsz 640 --fp16

# Repeat for: court_best.pt, ball_best.pt, bodypose_best.pt, shotclass_best.pt
```

Each `.engine` file runs at ~1–2 ms on Orin Nano at FP16.

---

## Post-Training Implementation (Days 5–7)

After all models finish training, the following modules need implementation
and integration:

| Module | File | Purpose |
|---|---|---|
| Ball tracker | `src/ball_tracker.py` | Kalman-filter smoothing of ball positions, bounce detection |
| Shot classifier | `src/shot_classifier.py` | Fuses shotclass detection + body pose features → refined shot type |
| Rules engine | `src/rules_engine.py` | State machine: IDLE→SERVING→RALLY→POINT_SCORED |
| Heatmap renderer | `src/heatmap.py` | Accumulate player foot positions → homography → KDE → overlay on court template |
| Multi-model analyzer | `src/analyzer.py` | Updates to PadelAnalyzer to orchestrate all 5 models per frame |
| TensorRT export | `src/export_trt.py` | Converts .pt → .engine with FP16 optimization |
| Jetson launcher | `scripts/launch_jetson.sh` | Docker/script to start the live analysis pipeline |

---

## Evaluation & Iteration Strategy

| Model | Primary metric | Secondary | Fallback if low |
|---|---|---|---|
| Detection | mAP50-95 > 0.80 | Precision > 0.90 | More epochs, higher imgsz (960), mosaic up to epoch 20 |
| Court pose | Keypoint mAP > 0.90 | – | Augment with synthetic court lines, increase imgsz |
| Ball detection | mAP50 > 0.85 | – | Train-only on Roboflow (balls are larger), then fine-tune on padelTracker100 |
| Body pose | Pose mAP > 0.85 | – | Higher imgsz (960), COCO pretrained (already using yolo26n-pose.pt) |
| Shot class | mAP50 > 0.90 | Per-class balanced accuracy | Class weights for rare classes (Salida de pared, Volea de reves), retrain with mixup 0.3 |

### Fallback plan if a model underperforms

1. Increase `imgsz` (next step: 960 with batch 4 for detection/pose)
2. Add more epochs (with `patience` increased or removed)
3. Adjust augmentation (stronger HSV, higher scale)
4. Data augmentation via synthesis (e.g., replicate rare shot classes)
5. Ensemble: run both the detection AND the underperforming model, fuse outputs

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **5 separate models** (not one multi-task model) | Independent training queues, independent iteration, independent TensorRT export. No risk of task interference. |
| **YOLOv26-nano for all** | Fits 4 GB VRAM. TensorRT FP16 on Orin Nano runs 5 models in <10 ms/frame total. |
| **Court: 26 keypoints** (not 10) | 26 named landmarks (cage+court+net+service) = zero ambiguity for cross-dataset merging. Overdetermined homography (26 pts) = more robust projection. Previous 10-point schema had unnamed indices → impossible to merge safely. |
| **Court v2 via pseudo-labeling** | After v1 trains on 788 annotated images, auto-label 3,615 Plaimaker images → retrain on ~4,400 images. Adds diverse camera angles without manual annotation. |
| **Body pose: full 100k frames** (no subsample) | 30 fps → every frame is nearly identical to the previous, but keeping all frames ensures the model sees every transition (swing, ball contact, follow-through). |
| **TrackNetV3 for ball (not YOLOv26)** | An 8 px ball at 1080p becomes 2.7 px at imgsz 640 — below YOLO's reliable detection threshold. TrackNetV3's heatmap regression with 8-frame temporal stacking was purpose-built for small fast balls. Sub-pixel precision enables bounce detection and trajectory analysis. |
| **Offline ReID (not online)** | Online ReID must be real-time, limiting feature quality. Offline DINOv2 clustering (non-real-time) gives authoritative 4-player IDs for stats, while BoT-SORT provides real-time fallback. |
| **Shot classification as detection model** (not classifier) | padelcnn/padel-ball-hit is an object detection dataset (bounding boxes + shot classes). Training a detection model on it detects both "a shot happened" AND "what type" in one pass. |
| **Court pose before body pose** | Court pose is fast (~1.5 h), enables inside-court filtering for ball detection (ignore balls outside court) and homography for the 2D map. |
| **TrackNetV3 last in queue** | TrackNetV3 uses a separate training pipeline (`src/train_tracknet.py`, not Ultralytics). It waits for all YOLO models to finish via `queue_ball_tracknet.sh`, then fine-tunes from shuttlecock-pretrained weights. |

---

## Files Reference

| File | Purpose |
|---|---|
| `configs/detection_combined.yaml` | Player detection training config |
| `configs/pose.yaml` | Court keypoint training config (26 kpts, 600 epochs) |
| `configs/ball.yaml` | TrackNetV3 ball config (heatmap regression, 8-frame stack) |
| `configs/bodypose.yaml` | Body pose training config (17 kpts, full 100k) |
| `configs/shotclass.yaml` | Shot classification training config (11 classes) |
| `configs/botsort_reid.yaml` | BoT-SORT + ReID shared config |
| `configs/inference.yaml` | Live inference pipeline config |
| `src/train.py` | Unified training entry point |
| `src/reid.py` | Offline K=4 Re-ID pipeline |
| `src/player_tracker.py` | CourtFilter, PlayerRegistry, PlayerTracker |
| `src/analyzer.py` | PadelAnalyzer — orchestrates models + ReID |
| `src/utils/homography.py` | 26-point reference template on 20×10 m court |
| `src/server/app.py` | FastAPI web server |
| `src/server/reid_resolver.py` | ReID caching + background compute |
| `src/server/modelutil.py` | Build analyzer with auto-loaded ReID |
| `src/server/training_api.py` | Training dashboard API (runs, metrics, dataset info) |
| `scripts/extract_frames.py` | Multi-process frame extraction (8 workers) |
| `scripts/convert_coco_pose_to_yolo.py` | COCO → YOLO-pose body keypoints |
| `scripts/convert_court_26.py` | Josh 26-class/1-kpt → 1-class/26-kpt conversion |
| `scripts/build_tracknet_dataset.py` | Convert padelTracker100 ball JSONs → TrackNetV3 rally format |
| `src/train_tracknet.py` | TrackNetV3 training script (separate from Ultralytics) |
| `src/ball_tracker.py` | TrackNetBallTracker inference module |
| `scripts/queue_ball_tracknet.sh` | TrackNetV3 ball training queue (after YOLO chain) |
| `third_party/tracknetv3/` | Vendored TrackNetV3 repo (model, loss, dataset) |
| `scripts/build_ball_dataset.py` | (Legacy) YOLOv26 ball dataset builder — unused |
| `scripts/queue_all_training.sh` | Unified training queue (ball→bodypose→shotclass) |
| `scripts/queue_court_training.sh` | Court pose queue (detection→pose) |
| `scripts/train_bodypose.sh` | Manual body-pose training launcher |
| `scripts/train_detection.sh` | Manual detection training launcher |
| `scripts/train_pose.sh` | Manual court-pose training launcher |
| `scripts/export_jetson.sh` | TensorRT export script |
| `data/datasets/combined/` | Player detection dataset (36,920 imgs) |
| `data/datasets/court_keypoints_26/` | Court keypoint dataset v1 (788 imgs, 26 kpts) |
| `data/datasets/court_26_josh/` | Raw Josh dataset (793 imgs, pre-conversion) |
| `data/datasets/court_plaimaker/` | Plaimaker images for pseudo-labeling (3,615 imgs) |
| `data/datasets/ball_tracknet/` | TrackNetV3 ball dataset (264 rally chunks) |
| `data/datasets/ball/` | (Legacy) YOLOv26 ball dataset — unused |
| `data/datasets/bodypose/` | Body pose dataset (99,822 imgs + symlinks) |
| `data/datasets/shotclass_roboflow/` | Shot classification dataset (8,959 imgs) |
| `data/datasets/padeltracker100/frames/` | Extracted match frames (99,887 JPG, 49 GB) |
| `data/models/` | Trained weight destination |
| `data/reid_cache/` | Cached ReID mapping JSONs |
| `docs/ROADMAP.md` | Original multi-model roadmap |
