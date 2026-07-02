# Roadmap — Padel Analytics (multi-model system)

Goal: one analysis system combining **player tracking + court→2D map + ball +
pose→shots**, mirroring Clutchapp.io. Each component is an independent model that
plugs into `src/analyzer.py` (PadelAnalyzer) and lights up the web UI feature by
feature. Architecture overview:

```
 frame ─┬─▶ PlayerTracker (YOLOv26 detect+track+reID) ─▶ tracked players (+team)
        ├─▶ CourtModel (YOLOv26-pose, 26 kpts) ────────▶ homography ─▶ 2D court map
         ├─▶ BallTracker (TrackNetV3) ────────────────────▶ ball position + trail
        ├─▶ PoseModel (YOLOv26-pose, 17 COCO kpts) ─────▶ stance ─▶ shot type
        └─▶ ShotClass (YOLOv26 detect, 11 classes) ─────▶ shot type per hit
                                                       └─▶ stats / heatmaps
```

> **Full training details:** see [`docs/model_training_architecture.md`](model_training_architecture.md)
> for configs, hyperparameters, dataset stats, and the training queue timeline.

---

## Component 1 — Player tracking (training, epoch 81/100)

- **Data:** Roboflow `octovar/player-padel-dataset` + **padelTracker100** (Zenodo) → combined **28,242 train / 8,678 val** (36,920 total).
- **Model:** YOLOv26-nano (detect), BoT-SORT tracking with persistent reID IDs.
- **Status:** training on combined dataset (`configs/detection_combined.yaml`, mosaic+mixup+copy_paste). Outputs `data/models/player_best.pt`.
- **ReID:** Offline DINOv2 K=4 clustering (`src/reid.py`) — done and tested.

## Component 2 — Court keypoints → 2D map (queued)

- **What:** detect 26 court landmarks (cage + court + net + service line) → compute overdetermined homography → project players/ball to a 20×10 m top-down court map.
- **Data:** Roboflow `joshs-workspace-p1aa0/padel-court-detection` → 788 images (596 train / 192 val), converted via `scripts/convert_court_26.py`.
- **Pseudo-labeling v2:** 3,615 Plaimaker court images → auto-label with v1 → retrain on ~4,400 images.
- **Model:** YOLOv26-nano-pose, `kpt_shape: [26, 3]`. Config: `configs/pose.yaml` (600 epochs, patience 80).
- **Code:** `src/utils/homography.py` (26-point reference template, compute_homography, project_points).
- **Status:** Queued after detection training.

## Component 3 — Ball detection (TrackNetV3, queued)

- **What:** per-frame ball center → trajectory trail, speed, bounce detection, rally state machine, scoring.
- **Why TrackNetV3 over a general YOLO-ball:** the ball is ~8 px at 1080p and moves fast, and we need ~45–60 fps live for accurate shot/contact detection. A YOLOv26 detector at imgsz 960 sustains only ~15–25 fps on Jetson Orin Nano — not enough. TrackNetV3 (CenterNet / hourglass backbone, gaussian heatmap regression) is purpose-built for small, fast balls and is the de-facto padel/tennis ball tracker.
- **Data:** Roboflow `padel/padel-ball-detector` (3,153 imgs) + padelTracker100 ball JSONs (subsampled 1/5 = 11,455 imgs) → combined **13,196 train / 1,233 val** (14,429 total). Point annotations are converted to CenterNet gaussian heatmap targets.
- **Model:** **TrackNetV3** — NOT a YOLO model (see training caveat below).
- **Training caveat:** TrackNetV3 cannot reuse `src/train.py` (Ultralytics-only). It needs:
  - `scripts/build_tracknet_dataset.py` — convert Roboflow + padelTracker100 ball points → heatmap targets.
  - `src/train_tracknet.py` — CenterNet training loop (separate from `src/train.py`).
  - `configs/ball.yaml` — to be rewritten from YOLOv26 to TrackNetV3 config.
  - `src/ball_tracker.py` — TrackNetV3 inference + Kalman smoothing/trail, replacing the stub at `src/analyzer.py:108`.
  - Outputs `data/models/ball_best.pt` (exportable to TensorRT).
- **Status:** Queued after body-pose training. Largest single component (~2–3× the work of the other models).

## Component 4 — Body pose → shot classification (queued)

- **What:** per-player 17 COCO body keypoints → stance/arm → fused with shot-class model for shot type classification (drive, lob, bandeja, víbora, smash, globo).
- **Body pose data:** padelTracker100 pose JSONs → 99,822 images (84,847 train / 14,975 val). Config: `configs/bodypose.yaml`.
- **Shot class data:** Roboflow `francisco-gonzalez-dkgry/padel-ball-hit` → 8,959 images, 11 shot-type classes. Config: `configs/shotclass.yaml`.
- **Fusion:** Shot-class detection + body pose keypoint features → MLP classifier for refined accuracy.
- **Status:** Body pose queued after ball detection. Shot class queued after body pose.

## Component 5 — Production / Jetson (live)

- Export each `.pt` → `.engine` **on the Jetson** (`src/export_trt.py`, TensorRT FP16). TrackNetV3 also exports to TensorRT.
- **Live multi-model cadence** (see main plan): player detect+track ~30 fps @640, ball ~45–60 fps (TrackNetV3), court keypoints every ~2 s / scene change, body pose every 3rd frame on 4 player crops, shotclass event-triggered (ball-in-racket-zone).
- Online Re-ID resolver (`src/reid_online.py`, to create): warmup (~30–60 s) → per-detection EMA assignment → periodic re-cluster → stable P1..P4 during the match; final full re-cluster post-match.
- Match lifecycle (`src/match_state.py`): **WARMUP → LIVE → ENDED**. Warmup is the Re-ID calibration window; stats count only during LIVE. Match starts on button-click OR serve auto-detect (whichever first), with the button authoritative — see [`docs/match_lifecycle.md`](match_lifecycle.md).
- Automatic team assignment: court-side split (P1+P2 vs P3+P4).
- Serve via web app (FastAPI MVP → potential DeepStream migration for multi-camera).
- Rules engine: `src/rules_engine.py` — **v1 = rally/point detection + shot attribution only (no scoring).** Full padel scoring is scoped in [`docs/scoring_spec.md`](scoring_spec.md).

> **Sustained-load validation — DEFERRED until on-field hardware is available.**
>
> The Orin Nano must be benchmarked under the *full live multi-model stack* for an entire ~90-minute match before the per-model cadence numbers above can be trusted. Jetson modules throttle under sustained load (thermal), which silently drops real throughput below the per-model estimates — the difference between "planning estimate" and "measured guarantee".
>
> The benchmark (to run during on-field testing week, once the 90 fps camera + Orin Nano are deployed) should measure:
> - sustained FPS per model over 90 min,
> - SoC temperature curve + thermal-throttling events,
> - GPU/NPU memory headroom,
> - end-to-end pipeline latency (capture → annotated frame).
>
> Until then, all fps targets in this roadmap are planning estimates, not measured guarantees.

---

## Design principle

Every model is a self-contained checkpoint consumed by a swappable module in
`PadelAnalyzer`. Swapping the tracker (Ultralytics→DeepStream) or the ball model
(YOLOv26→TrackNetV3) must not change the stats/analytics layer.
