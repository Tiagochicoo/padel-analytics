# Padel Analytics — Production CV Pipeline

A modular, production-oriented computer-vision system for padel game analysis
(inspired by Clutchapp.io). Built for training on a laptop GPU and deploying
to NVIDIA Jetson Orin Nano via TensorRT.

> Status: **Phase 1** — player detection/tracking + court keypoint (pose) detection.
> See `docs/ROADMAP.md` for TrackNetV3 (ball), homography, pose, stats & DeepStream.

---

## 1. Pipeline architecture

```
                 ┌────────────────────────┐
   Video / Camera│  YOLOv26 (detect+track)│ ── players + reID
        ────────▶│  YOLOv26-pose (keypt)  │ ── court corners (homography)
                 │  TrackNetV3 (ball)     │ ── ball trajectory  (Phase 2)
                 └───────────┬────────────┘
                             │ 2D image coords
                    ┌────────▼─────────┐
                    │   Homography H   │ ── project to top-down court
                    └────────┬─────────┘
                             │ real-world coords
                ┌────────────▼───────────────┐
                │ Stats / heatmaps / shots   │
                └────────────────────────────┘
```

Each stage is an independent, swappable module so models can be trained and
exported separately and dropped onto the Jetson.

---

## 2. Hardware / software target

| Role        | Device                                | Notes                         |
|-------------|---------------------------------------|-------------------------------|
| Training    | ThinkPad, RTX 3050 Ti (4 GB)          | Use `batch=8`, FP16, imgsz 640|
| Inference   | Jetson Orin Nano + TensorRT           | FP16 `.engine` built ON Jetson|

> ⚠️ TensorRT plan files (`.engine`) are tied to the GPU + TensorRT/CUDA version.
> They are **not portable**. Always build the engine on the final device (Jetson).
> See `src/export_trt.py`.

---

## 3. Project structure

```
padel-analytics/
├── README.md                      # this file
├── requirements.txt               # dev / training deps (laptop)
├── requirements-jetson.txt        # Jetson inference deps
├── .gitignore
├── .env.example                   # copy -> .env, add your Roboflow key
├── .vscode/
│   ├── settings.json
│   └── extensions.json
├── configs/
│   ├── detection.yaml             # data.yaml for player detection
│   ├── pose.yaml                  # data.yaml for court keypoints
│   └── inference.yaml             # runtime inference settings
├── data/
│   ├── datasets/                  # Roboflow downloads (gitignored)
│   │   ├── players/
│   │   └── court_keypoints/
│   ├── models/                    # trained/exported weights (gitignored)
│   └── sample_videos/
├── src/
│   ├── __init__.py
│   ├── download_data.py           # Roboflow downloader
│   ├── train.py                   # unified train entry point
│   ├── export_trt.py              # export -> ONNX / TensorRT
│   ├── infer.py                   # video + live camera inference
│   └── utils/
│       ├── __init__.py
│       ├── camera.py              # unified video/camera source
│       ├── visualization.py       # drawing helpers
│       └── homography.py          # court homography (Phase 2 stub)
├── scripts/
│   ├── train_detection.sh
│   ├── train_pose.sh
│   └── export_jetson.sh
├── notebooks/                     # exploratory analysis
└── docs/
    └── ROADMAP.md
```

---

## 4. Setup (step-by-step)

### 4.1 Clone / open the project in VS Code
```bash
code /home/tpereira/rep/padel-analytics
```
VS Code will pick up `.vscode/` (recommended extensions + interpreter).

### 4.2 Get a compatible Python (important!)
Your system ships **Python 3.14**, which is too new for PyTorch wheels.
Use **Python 3.11 or 3.12**. The cleanest way is `uv` (fast, fetches Python):

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# create a venv with Python 3.11 inside the project
cd /home/tpereira/rep/padel-analytics
uv venv --python 3.11 .venv
source .venv/bin/activate
```

(Alternative: `sudo apt install python3.12-venv && python3.12 -m venv .venv`.)

### 4.3 Install dependencies
```bash
# make sure pip is up to date
uv pip install --upgrade pip

# PyTorch with CUDA (matches your CUDA toolkit; adjust cu121/cu124 as needed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# project deps
uv pip install -r requirements.txt
```

Verify GPU access:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.4 Configure Roboflow credentials
```bash
cp .env.example .env
# edit .env and paste your ROBOFLOW_API_KEY + workspace/project/version IDs
```

### 4.5 Download datasets
```bash
python src/download_data.py --task detection
python src/download_data.py --task pose
```

### 4.6 Train
```bash
bash scripts/train_detection.sh    # players
bash scripts/train_pose.sh         # court keypoints
```

### 4.7 Inference (video or live camera)
```bash
# video file
python src/infer.py --model detection --weights data/models/player_best.pt --source data/sample_videos/match.mp4

# live camera (index 0) or RTSP
python src/infer.py --model detection --weights data/models/player_best.pt --source 0
python src/infer.py --model detection --weights data/models/player_best.pt --source rtsp://user:pass@ip/stream
```

---

## 5. Model versions note

The pipeline targets Ultralytics **YOLOv26**. Weight names follow
`yolo26n.pt` / `yolo26n-pose.pt`. These are set once in the `configs/*.yaml`
(`base` field) and the shell scripts. If a given version isn't yet released,
simply change that one field to the latest available Ultralytics model
(e.g. `yolo11n.pt`) — the rest of the code is version-agnostic.

---

## 6. Moving models to Jetson

1. Train on the laptop → copy `runs/.../weights/best.pt` to the Jetson.
2. On the Jetson (JetPack SDK, with TensorRT), export the engine:
   ```bash
   python src/export_trt.py --weights best.pt --format engine --half --imgsz 640
   ```
3. Run inference with the `.engine` (see `scripts/export_jetson.sh`).

See `docs/ROADMAP.md` for the DeepStream migration plan.
