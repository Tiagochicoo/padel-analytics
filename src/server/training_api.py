"""
training_api.py
===============
Read-only access to Ultralytics training runs for the web dashboard.

Scans runs/detect (and runs/pose) for run directories, parses each run's
``results.csv`` (per-epoch metrics) + ``args.yaml`` (config), detects which run
is currently active (a live ``src/train.py`` process) and tails the training
log for a live feed. Used by the /training page.
"""

from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"

# Columns we surface in the dashboard (csv header -> friendly key).
METRIC_COLS = {
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
}

# Config fields worth showing.
ARG_FIELDS = ["task", "model", "data", "epochs", "batch", "imgsz", "device",
              "optimizer", "lr0", "cos_lr", "close_mosaic", "mosaic", "mixup",
              "copy_paste", "patience", "name", "save_dir"]


def _run_dirs() -> list[Path]:
    if not RUNS_ROOT.exists():
        return []
    dirs = []
    for task_dir in sorted(RUNS_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.iterdir()):
            # Include a run as soon as it starts (args.yaml written at launch),
            # so a brand-new run shows at epoch 0 before results.csv exists.
            if run_dir.is_dir() and (
                (run_dir / "results.csv").exists() or (run_dir / "args.yaml").exists()
            ):
                dirs.append(run_dir)
    return dirs


def _run_mtime(run_dir: Path) -> float:
    """Most recent touch across the files a trainer writes (for active-run detection)."""
    return max(
        _mtime(run_dir / "args.yaml"),
        _mtime(run_dir / "results.csv"),
        _mtime(run_dir / "weights" / "last.pt"),
    )


# Long enough for the slowest model (bodypose: ~59 min/epoch at 100k images).
_ACTIVE_STALENESS_S = 5400  # 90 minutes


def _is_run_active(proc: Optional[str], run_args: dict, mtime: float) -> bool:
    """Determine whether a specific run is the one currently training.

    Primary check: match the running process's ``--task`` to the run's
    ``data`` config (e.g. proc="--task bodypose" ↔ data="configs/bodypose.yaml").
    Falls back to a staleness check on ``last.pt`` mtime.
    """
    if not proc:
        return False

    # Extract --task value from the process cmdline
    # e.g. "python src/train.py --task bodypose" → "bodypose"
    task = None
    for part in proc.replace(b"\x00", b" ").split() if isinstance(proc, bytes) else proc.split():
        if part == "--task":
            task = ""  # next part is the value
        elif task == "":
            task = part
            break

    if task:
        # Match task name against the run's data config path
        data_cfg = str(run_args.get("data", "")).lower()
        # "bodypose" in "configs/bodypose.yaml" → match
        # "pose" in "configs/pose.yaml" → match (but "bodypose" contains "pose")
        # So check exact config filename: configs/<task>.yaml
        if f"{task}.yaml" in data_cfg or f"/{task}." in data_cfg:
            return True

    # Fallback: staleness check (any run recently touched)
    return (time.time() - mtime) < _ACTIVE_STALENESS_S


def _read_args(run_dir: Path) -> dict:
    p = run_dir / "args.yaml"
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}
    return {k: data.get(k) for k in ARG_FIELDS if k in data}


# ---------------------------------------------------------------------------
# Dataset image counts (cached — scanning 30k files every poll is wasteful).
# ---------------------------------------------------------------------------
_dataset_cache: dict[str, tuple[float, dict]] = {}
_DATASET_TTL = 60.0  # seconds


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    n = 0
    try:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
                n += 1
    except OSError:
        pass
    return n


def _resolve_split(path_val: str, split_val: str) -> Optional[Path]:
    """Resolve the (path, split) pair from a data yaml to a directory."""
    base = Path(path_val) if path_val else PROJECT_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    split = Path(split_val) if split_val else None
    if split is None:
        return None
    return split if split.is_absolute() else base / split


def load_dataset_info(data_path: str) -> dict:
    """Read a data/config yaml and count train/val images. Cached 60s.

    The ``data`` field in a run's ``args.yaml`` points at a config yaml
    (e.g. ``configs/detection_combined.yaml``) which carries Ultralytics'
    ``path``/``train``/``val``/``nc``/``names`` keys.
    """
    if not data_path:
        return {}
    p = Path(data_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    key = str(p)
    now = time.time()
    cached = _dataset_cache.get(key)
    if cached and (now - cached[0]) < _DATASET_TTL:
        return cached[1]
    info: dict = {}
    if not p.exists():
        _dataset_cache[key] = (now, info)
        return info
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        _dataset_cache[key] = (now, info)
        return info
    info["nc"] = data.get("nc")
    names = data.get("names")
    if isinstance(names, dict):
        info["names"] = list(names.values())
    elif isinstance(names, list):
        info["names"] = names
    path_val = str(data.get("path", "") or "")
    train_dir = _resolve_split(path_val, str(data.get("train", "") or ""))
    val_dir = _resolve_split(path_val, str(data.get("val", "") or ""))
    info["train_count"] = _count_images(train_dir) if train_dir else 0
    info["val_count"] = _count_images(val_dir) if val_dir else 0
    info["total"] = info["train_count"] + info["val_count"]
    _dataset_cache[key] = (now, info)
    return info


def _read_csv(run_dir: Path) -> list[dict]:
    p = run_dir / "results.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            r = {"epoch": int(float(raw.get("epoch", 0)))}
            for col, key in METRIC_COLS.items():
                v = raw.get(col)
                if v is None or v == "":
                    r[key] = None
                else:
                    try:
                        r[key] = float(v)
                    except ValueError:
                        r[key] = None
            # Ultralytics writes a cumulative wall-clock "time" column (seconds
            # since training start). We keep it to derive per-epoch averages and
            # remaining-time estimates downstream.
            tv = raw.get("time")
            try:
                r["time"] = float(tv) if tv not in (None, "") else None
            except ValueError:
                r["time"] = None
            rows.append(r)
    return rows


def _training_process_running() -> Optional[str]:
    """Return the cmdline of a live src/train.py process, or None."""
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmd = (entry / "cmdline").read_bytes()
            except (OSError, PermissionError):
                continue
            if not cmd:
                continue
            text = cmd.replace(b"\x00", b" ").decode("utf-8", "ignore")
            if "src/train.py" in text and "--task" in text:
                return text.strip()
    except Exception:
        pass
    return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _log_path_for(run_dir: Path) -> Optional[Path]:
    """Map a run directory to its training log file."""
    # Read the run's args to determine which task produced it
    args = _read_args(run_dir)
    data_cfg = str(args.get("data", ""))
    task_dir = run_dir.parent.name  # "detect", "pose", "tracknet"

    # Map task/data config to log file
    if "detection" in data_cfg or "combined" in data_cfg:
        log = PROJECT_ROOT / "data" / "training_combined.log"
    elif "bodypose" in data_cfg:
        log = PROJECT_ROOT / "data" / "training_bodypose.log"
    elif "shotclass" in data_cfg:
        log = PROJECT_ROOT / "data" / "training_shotclass.log"
    elif "pose" in data_cfg or task_dir == "pose":
        log = PROJECT_ROOT / "data" / "training_pose.log"
    elif task_dir == "tracknet":
        log = PROJECT_ROOT / "data" / "training_ball.log"
    else:
        log = PROJECT_ROOT / "data" / "training.log"

    return log if log.exists() else None


def _per_epoch_seconds(rows: list[dict]) -> Optional[float]:
    """Median seconds-per-epoch derived from the CSV cumulative ``time`` column.

    Ultralytics writes a running wall-clock total, so each epoch's duration is
    the delta between consecutive rows. We take the median of the most recent
    deltas for a stable estimate that ignores early-epoch warmup spikes.
    """
    deltas = []
    prev = None
    for r in rows:
        t = r.get("time")
        if t is None:
            continue
        if prev is not None and t > prev:
            deltas.append(t - prev)
        prev = t
    if not deltas:
        return None
    deltas = sorted(deltas)
    return deltas[len(deltas) // 2]


# tqdm-style progress line, e.g.:
#   81/100  1.71G 0.883 0.358 0.0035  49  640: 31% ━━╸── 1128/3531 6.8it/s 3:08<5:56
_PROGRESS_RE = re.compile(
    r"(\d+)/(\d+)\b.*?\b(\d+):\s*(\d+)%.*?(\d+)/(\d+)\b\s+"
    r"([\d.]+)\s*it/s\s+(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)"
)


def _parse_progress(log_lines: list[str]) -> Optional[dict]:
    """Extract the most recent in-epoch tqdm progress from a log tail.

    Returns current_epoch, total_epochs, batches_done, batches_total, the
    iteration rate, elapsed seconds within the current epoch, and the epoch's
    remaining seconds. Returns None if no progress line is found.
    """
    for ln in reversed(log_lines):
        m = _PROGRESS_RE.search(ln)
        if not m:
            continue
        ep_cur, ep_tot = int(m.group(1)), int(m.group(2))
        b_done, b_tot = int(m.group(5)), int(m.group(6))
        rate = float(m.group(7))
        elapsed = _parse_hms(m.group(8))
        remaining = _parse_hms(m.group(9))
        return {
            "epoch": ep_cur,
            "total_epochs": ep_tot,
            "batches_done": b_done,
            "batches_total": b_tot,
            "it_per_s": rate,
            "epoch_elapsed_sec": elapsed,
            "epoch_remaining_sec": remaining,
            "batch_progress": round(b_done / b_tot, 4) if b_tot else 0.0,
        }
    return None


def _parse_hms(s: str) -> int:
    """'3:08' or '1:02:03' -> total seconds."""
    parts = [int(p) for p in s.split(":") if p != ""]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def estimate_timing(run_dir: Path, rows: list[dict], total: int,
                    active: bool, log_tail: list[str]) -> dict:
    """Compute training-time estimates for the dashboard.

    Combines three signals:
      * per-epoch average from the CSV cumulative ``time`` column,
      * the in-epoch tqdm line (current-epoch %, rate, remaining),
      * the weights/last.pt mtime as the wall-clock epoch boundary.
    """
    out: dict = {}
    if not rows:
        return out
    per_epoch = _per_epoch_seconds(rows)
    epochs_done = rows[-1]["epoch"]
    out["per_epoch_sec"] = round(per_epoch, 1) if per_epoch else None
    out["epochs_done"] = epochs_done
    out["epochs_total"] = total

    last = rows[-1].get("time")
    out["elapsed_sec"] = round(last, 1) if last else None

    remaining_epochs = max(0, total - epochs_done)
    base_remaining = (per_epoch or 0) * remaining_epochs

    prog = _parse_progress(log_tail) if active else None
    out["current_epoch"] = prog["epoch"] if prog else (epochs_done + 1 if active and total else None)
    out["current_epoch_progress"] = prog["batch_progress"] if prog else None
    out["current_epoch_remaining_sec"] = prog["epoch_remaining_sec"] if prog else None
    out["it_per_s"] = round(prog["it_per_s"], 2) if prog else None

    # If we have a live progress line, the current epoch is already partly done,
    # so subtract the proportional slice of one epoch's budget from the total.
    total_remaining = base_remaining
    if prog and per_epoch:
        # current epoch counted fully in base_remaining; credit the elapsed part
        total_remaining = base_remaining - prog["epoch_elapsed_sec"]
        # and prefer the tqdm estimate for the rest of this epoch
        total_remaining = total_remaining - (per_epoch - prog["epoch_remaining_sec"]) + prog["epoch_remaining_sec"]
        total_remaining = prog["epoch_remaining_sec"] + per_epoch * max(0, remaining_epochs - 1)
    out["total_remaining_sec"] = round(total_remaining, 1) if total_remaining else None

    # Absolute ETA from the last checkpoint mtime (best wall-clock anchor).
    last_pt = run_dir / "weights" / "last.pt"
    anchor = _mtime(last_pt) or _mtime(run_dir / "results.csv")
    if active and anchor and total_remaining:
        out["eta_at"] = anchor + total_remaining
    return out


def _tail(path: Path, n: int = 40) -> list[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    # split keeping only complete lines, strip ANSI escapes
    lines = data.split(b"\n")
    tail = lines[-n:]
    ansi = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")
    out = []
    for ln in tail:
        ln = ansi.sub(b"", ln).decode("utf-8", "ignore").rstrip()
        ln = re.sub(r"\r", "", ln)
        if ln.strip():
            out.append(ln)
    return out


def list_runs() -> list[dict]:
    proc = _training_process_running()
    runs = []
    for rd in _run_dirs():
        rows = _read_csv(rd)
        last_pt = rd / "weights" / "last.pt"
        mtime = _run_mtime(rd)
        args = _read_args(rd)
        active = _is_run_active(proc, args, mtime)
        epochs_done = rows[-1]["epoch"] if rows else 0
        total = int(args.get("epochs", 0) or 0) or 0
        latest = rows[-1] if rows else {}
        patience = int(args.get("patience", 0) or 0)
        last_exists = (rd / "weights" / "last.pt").exists()
        if active:
            _status = "training"
        elif total and epochs_done >= total:
            _status = "done"
        elif patience and epochs_done >= patience and last_exists:
            _status = "done"
        elif epochs_done > 0:
            _status = "crashed"
        else:
            _status = "pending"
        runs.append({
            "id": rd.name,
            "task": rd.parent.name,
            "path": str(rd),
            "active": active,
            "status": _status,
            "epochs_done": epochs_done,
            "epochs_total": total,
            "progress": round(epochs_done / total, 3) if total else 0.0,
            "model": args.get("model"),
            "data": args.get("data"),
            "last_mAP50": latest.get("mAP50"),
            "last_mAP50_95": latest.get("mAP50_95"),
            "mtime": mtime,
            "dataset_total": load_dataset_info(args.get("data", "")).get("total", 0),
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def get_run(run_id: str) -> Optional[dict]:
    rd = None
    for d in _run_dirs():
        if d.name == run_id:
            rd = d
            break
    if rd is None:
        return None
    rows = _read_csv(rd)
    args = _read_args(rd)
    last_pt = rd / "weights" / "last.pt"
    best_pt = rd / "weights" / "best.pt"
    proc = _training_process_running()
    mtime = _run_mtime(rd)
    active = _is_run_active(proc, args, mtime)
    epochs_done = rows[-1]["epoch"] if rows else 0
    total = int(args.get("epochs", 0) or 0) or 0
    patience = int(args.get("patience", 0) or 0)
    last_exists = last_pt.exists()
    if active:
        _status = "training"
    elif total and epochs_done >= total:
        _status = "done"
    elif patience and epochs_done >= patience and last_exists:
        _status = "done"
    elif epochs_done > 0:
        _status = "crashed"
    else:
        _status = "pending"
    logp = _log_path_for(rd)
    log_tail = _tail(logp, 60) if logp else []
    return {
        "id": rd.name,
        "task": rd.parent.name,
        "path": str(rd),
        "args": args,
        "active": active,
        "status": _status,
        "process": proc,
        "epochs_done": epochs_done,
        "epochs_total": total,
        "progress": round(epochs_done / total, 3) if total else 0.0,
        "has_best": best_pt.exists(),
        "has_last": last_pt.exists(),
        "series": rows,
        "log_tail": log_tail,
        "dataset_info": load_dataset_info(args.get("data", "")),
        "timing": estimate_timing(rd, rows, total, active, log_tail),
    }


# ---------------------------------------------------------------------------
# Planned / queued training configs — scan configs/*.yaml to surface upcoming
# training entries on the dashboard even before a run directory is created.
# ---------------------------------------------------------------------------

# Config stem -> expected output model filename (mirrors src/train.py weight_names)
_CONFIG_OUTPUT = {
    "detection_combined": "player_best.pt",
    "detection_combined_s": "player_s_best.pt",
    "pose": "court_best.pt",
    "pose_s": "court_s_best.pt",
    "bodypose": "bodypose_best.pt",
    "bodypose_s": "bodypose_s_best.pt",
    "shotclass": "shotclass_best.pt",
    "shotclass_s": "shotclass_s_best.pt",
}

_CONFIG_LABELS = {
    "detection_combined": "Player Detection",
    "detection_combined_s": "Player Detection (s)",
    "pose": "Court Keypoints",
    "pose_s": "Court Keypoints (s)",
    "bodypose": "Body Pose",
    "bodypose_s": "Body Pose (s)",
    "shotclass": "Shot Classification",
    "shotclass_s": "Shot Classification (s)",
}


def _is_config_running(proc: Optional[str], config_stem: str) -> bool:
    """Check if a live src/train.py process matches this config stem."""
    if not proc:
        return False
    needle = f"--task {config_stem}"
    return needle in proc


def list_planned(proc: Optional[str] = None) -> list[dict]:
    """Scan configs/ for training configs and determine their status.

    Returns a list of dicts with: id, label, base, epochs, batch, imgsz,
    output_model, model_exists, status (done|running|queued), is_yolo.
    """
    configs_dir = PROJECT_ROOT / "configs"
    models_dir = PROJECT_ROOT / "data" / "models"
    planned: list[dict] = []

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(cfg_path.read_text())
        except Exception:
            continue
        # Only YOLO training configs have base + hyp
        if not data.get("base") or not data.get("hyp"):
            continue

        stem = cfg_path.stem
        hyp = data.get("hyp", {})
        model_name = _CONFIG_OUTPUT.get(stem, f"{stem}_best.pt")
        model_path = models_dir / model_name

        if model_path.exists():
            status = "done"
        elif _is_config_running(proc, stem):
            status = "running"
        else:
            status = "queued"

        ds_info = load_dataset_info(str(cfg_path))
        planned.append({
            "id": stem,
            "label": _CONFIG_LABELS.get(stem, stem),
            "base": data.get("base", ""),
            "epochs": hyp.get("epochs"),
            "batch": hyp.get("batch"),
            "imgsz": hyp.get("imgsz"),
            "output_model": model_name,
            "model_exists": model_path.exists(),
            "status": status,
            "is_yolo": True,
            "dataset_total": ds_info.get("total", 0),
        })

    # TrackNetV3 ball (special — not a YOLO config, hardcoded metadata)
    ball_model = models_dir / "ball_best.pt"
    ball_cfg = configs_dir / "ball.yaml"
    ball_ds = 0
    if ball_cfg.exists():
        try:
            ball_data = yaml.safe_load(ball_cfg.read_text())
            ball_ds = ball_data.get("data", {})
        except Exception:
            pass
    planned.append({
        "id": "ball",
        "label": "Ball Tracking (TrackNetV3)",
        "base": "TrackNetV3 (shuttlecock pretrained)",
        "epochs": 30,
        "batch": 8,
        "imgsz": "288×512",
        "output_model": "ball_best.pt",
        "model_exists": ball_model.exists(),
        "status": "done" if ball_model.exists() else "queued",
        "is_yolo": False,
        "dataset_total": 264,
    })

    order = {"running": 0, "queued": 1, "done": 2}
    planned.sort(key=lambda x: (order.get(x["status"], 3), x["id"]))
    return planned


def overview() -> dict:
    """Top-level status for the dashboard header."""
    proc = _training_process_running()
    runs = list_runs()
    active = next((r for r in runs if r["active"]), None)
    return {
        "training_running": proc is not None,
        "process": proc,
        "active_run": active,
        "runs": runs,
        "planned": list_planned(proc),
    }


def run_dir_for(run_id: str) -> Optional[Path]:
    """Resolve a run id (folder name) to its run directory, or None."""
    for d in _run_dirs():
        if d.name == run_id:
            return d
    return None


_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}

# prefix -> friendly category shown in the gallery
def _image_category(stem: str) -> str:
    s = stem.lower()
    if s.startswith("train_batch"):
        return "train"
    if s.startswith("val_batch"):
        if "pred" in s:
            return "predictions"
        if "label" in s:
            return "val_labels"
        return "val"
    if s.startswith("labels"):
        return "dataset"
    for p in ("results", "confusion_matrix", "pr_curve", "f1_curve",
              "p_curve", "r_curve"):
        if s.startswith(p):
            return "chart"
    return "other"


def list_run_images(run_id: str) -> list[dict]:
    """List the preview/plot images a run has written to disk."""
    rd = run_dir_for(run_id)
    if rd is None:
        return []
    out = []
    for p in sorted(rd.iterdir()):
        if p.suffix.lower() not in _IMG_SUFFIXES:
            continue
        out.append({
            "name": p.name,
            "category": _image_category(p.stem),
            "mtime": int(_mtime(p)),
            "size": p.stat().st_size if p.is_file() else 0,
        })
    return out


# ---------------------------------------------------------------------------
# Annotated dataset samples — pull random training images directly from the
# dataset and render them with YOLO annotations overlaid.  This gives the
# gallery far more variety than the 3-4 batch images Ultralytics writes.
# ---------------------------------------------------------------------------

# Cache the file listing for large datasets (84k+ files).
_sample_cache: dict[str, tuple[float, list[str]]] = {}
_SAMPLE_TTL = 300.0  # 5 min


def _resolve_train_dir(run_id: str) -> Optional[Path]:
    """Resolve the training image directory for a run from its args.yaml."""
    rd = run_dir_for(run_id)
    if rd is None:
        return None
    args = _read_args(rd)
    data_cfg_path = args.get("data")
    if not data_cfg_path:
        return None
    p = Path(data_cfg_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text())
    except Exception:
        return None
    return _resolve_split(str(data.get("path", "")), str(data.get("train", "")))


def _list_dataset_files(run_id: str) -> list[str]:
    """Cached listing of training image filenames for a run."""
    now = time.time()
    cached = _sample_cache.get(run_id)
    if cached and (now - cached[0]) < _SAMPLE_TTL:
        return cached[1]
    train_dir = _resolve_train_dir(run_id)
    if not train_dir or not train_dir.is_dir():
        _sample_cache[run_id] = (now, [])
        return []
    files = sorted(p.name for p in train_dir.iterdir()
                   if p.suffix.lower() in _IMG_SUFFIXES)
    _sample_cache[run_id] = (now, files)
    return files


def list_dataset_samples(run_id: str, n: int = 20) -> dict:
    """Pick *n* random training images for the gallery.

    Uses a seed derived from run_id so the same set is returned within the
    cache window (avoids shuffling every 5-second poll).
    """
    import random
    files = _list_dataset_files(run_id)
    if not files:
        return {"samples": [], "total": 0}
    rng = random.Random(hash(run_id) & 0xFFFFFFFF)
    n = min(n, len(files))
    picks = rng.sample(files, n)
    return {"samples": picks, "total": len(files)}


# COCO-17 skeleton connections
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# BGR colors for skeleton joints/limbs
_KPT_COLOR = (0, 255, 255)    # cyan dots
_LIMB_COLOR = (255, 200, 0)   # blue-orange lines
_BOX_COLOR = (0, 255, 0)      # green boxes


def _read_yolo_labels(label_path: Path) -> list[list[float]]:
    """Read a YOLO label file → list of rows (each row is a list of floats)."""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append([float(x) for x in parts])
    return rows


def render_annotated_sample(run_id: str, filename: str,
                            max_dim: int = 960) -> Optional[bytes]:
    """Load a training image, draw its YOLO annotations, return JPEG bytes."""
    import cv2

    train_dir = _resolve_train_dir(run_id)
    if not train_dir:
        return None
    img_path = train_dir / filename
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # Downscale if very large (1080p → max_dim wide) to keep response fast
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    # Find label file (same stem, .txt, in labels/ sibling of images/)
    # YOLO layout: dataset/images/train  →  dataset/labels/train
    label_dir = train_dir.parent.parent / "labels" / train_dir.name
    label_path = label_dir / (Path(filename).stem + ".txt")
    rows = _read_yolo_labels(label_path)

    is_pose = len(rows) > 0 and len(rows[0]) > 5  # has keypoint data

    for row in rows:
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        if is_pose:
            kpts = row[5:]
            n_kpt = len(kpts) // 3
            pts = []
            for i in range(n_kpt):
                kx, ky, kv = kpts[i * 3], kpts[i * 3 + 1], kpts[i * 3 + 2]
                px = int(kx * w)
                py = int(ky * h)
                pts.append((px, py, int(kv)))

            # Draw skeleton limbs
            for a, b in _COCO_SKELETON:
                if a < n_kpt and b < n_kpt:
                    if pts[a][2] > 0 and pts[b][2] > 0:
                        cv2.line(img, pts[a][:2], pts[b][:2],
                                 _LIMB_COLOR, 2, cv2.LINE_AA)
            # Draw keypoint dots
            for px, py, kv in pts:
                if kv > 0:
                    cv2.circle(img, (px, py), 4, _KPT_COLOR, -1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None
