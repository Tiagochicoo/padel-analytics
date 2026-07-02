#!/usr/bin/env python3
"""
build_shotclass_combined.py
===========================
Build a combined shot-classification dataset from two sources:

1. padelTracker100 shot CSVs (frame-level labels) + pose JSONs (ground-truth
   person bounding boxes + 17 COCO keypoints).  The hitter is identified
   from pose data (wrist-above-shoulder heuristic) — no model inference needed.

2. Roboflow padel-ball-hit (existing YOLO detection annotations, remapped
   from 11 classes to the simplified 6-class taxonomy).

Output: data/datasets/shotclass_combined/
    images/{train,val}/   — symlinks (no disk duplication)
    labels/{train,val}/   — YOLO format (class cx cy w h)
    data.yaml

Usage:
    python scripts/build_shotclass_combined.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────
PT100_RAW    = PROJECT_ROOT / "data/datasets/padeltracker100/raw"
PT100_FRAMES = PROJECT_ROOT / "data/datasets/padeltracker100/frames"
ROBOFLOW_DIR = PROJECT_ROOT / "data/datasets/shotclass_roboflow"
OUTPUT_DIR   = PROJECT_ROOT / "data/datasets/shotclass_combined"

# ── 9-class taxonomy ────────────────────────────────────────────────────
CLASS_NAMES = ["Forehand", "Backhand", "Smash", "Serve", "Volley", "Lob", "Wall shot", "Other", "Dropshot"]
NC = len(CLASS_NAMES)

# padelTracker100 category → combined class ID
PT100_MAP = {
    "Forehand": 0,
    "Backhand": 1,
    "Smash": 2,
    "Serve": 3,
    "Other": 7,
    "Dropshot": 8,
}

# Roboflow 11-class → combined 9-class
# None = drop annotation entirely (Bola is not a shot type)
ROBOFLOW_MAP = {
    0: 7,    # Bandeja → Other (too few for own class)
    1: None, # Bola → DROP (not a shot type)
    2: 6,    # Contrapared → Wall shot
    3: 5,    # Globo → Lob
    4: 0,    # Golpe de derecha → Forehand
    5: 1,    # Golpe de reves → Backhand
    6: 2,    # Remate → Smash
    7: 6,    # Salida de pared → Wall shot
    8: 3,    # Saque → Serve
    9: 4,    # Volea de derecha → Volley
    10: 4,   # Volea de reves → Volley
}

# Shot CSV → video prefix
SHOT_CSVS = {
    "2022_BCN_FinalF_1_shots.csv": "FinalF",
    "2022_BCN_FinalM_1_shots.csv": "FinalM",
}

# COCO keypoint indices (0-based, flat array stride=3)
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_WRIST = 9
KP_R_WRIST = 10

SUBSAMPLE_EVERY = 3       # take every Nth frame within a shot sequence
TEMPORAL_SPLIT = 0.85     # first 85% of each video → train
SEED = 42


# ── Step 1: Parse shot CSVs into sequences ──────────────────────────────

def parse_shot_csvs() -> list[list[tuple[str, int, str]]]:
    """Return list of shot sequences.  Each sequence is a list of
    (video_prefix, frame_index, category) tuples for consecutive frames."""
    sequences: list[list[tuple[str, int, str]]] = []
    for csv_name, prefix in SHOT_CSVS.items():
        csv_path = PT100_RAW / "labels" / csv_name
        if not csv_path.exists():
            print(f"  [warn] {csv_name} not found, skipping")
            continue
        shots: list[tuple[str, int, str]] = []
        with open(csv_path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row["has_shot"] != "1":
                    continue
                fname = row["file_name"]              # frame_000000.PNG
                idx = int(fname.replace("frame_", "").replace(".PNG", ""))
                shots.append((prefix, idx, row["category"]))
        # group consecutive frames with same category into sequences
        if not shots:
            continue
        current = [shots[0]]
        for i in range(1, len(shots)):
            prev_p, prev_i, prev_c = shots[i - 1]
            cur_p, cur_i, cur_c = shots[i]
            if cur_p == prev_p and cur_i == prev_i + 1 and cur_c == prev_c:
                current.append(shots[i])
            else:
                sequences.append(current)
                current = [shots[i]]
        sequences.append(current)
    return sequences


def subsample_sequences(
    sequences: list[list[tuple[str, int, str]]],
) -> list[tuple[str, int, str]]:
    """Take every Nth frame from each sequence.
    Rare classes (Dropshot) are NOT subsampled — keep all frames."""
    picks: list[tuple[str, int, str]] = []
    for seq in sequences:
        category = seq[0][2]
        # Don't subsample rare classes
        step = 1 if category == "Dropshot" else SUBSAMPLE_EVERY
        for i in range(0, len(seq), step):
            picks.append(seq[i])
    return picks


# ── Step 2: Load pose JSONs ─────────────────────────────────────────────

def load_pose_index() -> dict[str, dict[int, list[dict]]]:
    """Build index: {video_prefix: {frame_index: [annotation, ...]}}.
    Each annotation has 'bbox' [x,y,w,h] and 'keypoints' [x,y,v,...]."""
    pose_files = {
        "FinalF": PT100_RAW / "labels" / "2022_BCN_FinalF_1_pose.json",
        "FinalM": PT100_RAW / "labels" / "2022_BCN_FinalM_1_pose.json",
    }
    index: dict[str, dict[int, list[dict]]] = {}
    for prefix, path in pose_files.items():
        if not path.exists():
            print(f"  [warn] {path.name} not found")
            continue
        print(f"  loading {path.name}...")
        with open(path) as f:
            data = json.load(f)
        # image_id → frame_index
        id_to_frame: dict[int, int] = {}
        for img in data["images"]:
            fname = img["file_name"]  # frame_000000.PNG
            id_to_frame[img["id"]] = int(fname.replace("frame_", "").replace(".PNG", ""))
        # frame_index → annotations
        frame_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            frame_idx = id_to_frame.get(ann["image_id"])
            if frame_idx is not None:
                frame_anns[frame_idx].append(ann)
        index[prefix] = dict(frame_anns)
        print(f"    {len(data['images'])} images, {len(data['annotations'])} annotations")
    return index


# ── Step 3: Identify hitter from pose keypoints ─────────────────────────

def identify_hitter(annotations: list[dict]) -> dict | None:
    """Find the person whose wrist is highest above their shoulder.
    Falls back to most-visible-keypoints person if no clear swing."""
    best_score = -1e9
    best_ann = None
    best_kpt_count = -1
    best_fallback = None

    for ann in annotations:
        kpts = ann.get("keypoints", [])
        if len(kpts) < 51:  # 17 * 3
            continue

        def _y(idx):
            return kpts[idx * 3 + 1]

        def _v(idx):
            return kpts[idx * 3 + 2]

        # count visible keypoints for fallback
        vis_count = sum(1 for i in range(17) if _v(i) >= 2)
        if vis_count > best_kpt_count:
            best_kpt_count = vis_count
            best_fallback = ann

        # swing score: shoulder_y - wrist_y (positive = wrist above shoulder)
        shoulders, wrists = [], []
        for ki in (KP_L_SHOULDER, KP_R_SHOULDER):
            if _v(ki) >= 2:
                shoulders.append(_y(ki))
        for ki in (KP_L_WRIST, KP_R_WRIST):
            if _v(ki) >= 2:
                wrists.append(_y(ki))

        if shoulders and wrists:
            shoulder_y = min(shoulders)  # highest shoulder
            wrist_y = min(wrists)        # highest wrist
            score = shoulder_y - wrist_y
            if score > best_score:
                best_score = score
                best_ann = ann

    # If swing score is marginal (< 5px), use fallback
    if best_ann is not None and best_score > 5:
        return best_ann
    return best_fallback


# ── Step 4: YOLO label conversion ───────────────────────────────────────

def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, w, h] (absolute, top-left) to YOLO [cx, cy, w, h] (normalised)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


# ── Step 5: Process Roboflow data ───────────────────────────────────────

def process_roboflow(out_train_img: Path, out_val_img: Path,
                     out_train_lbl: Path, out_val_lbl: Path) -> Counter:
    """Copy Roboflow images + remapped labels to the output directory."""
    counts = Counter()
    for split, out_img_dir, out_lbl_dir in [
        ("train", out_train_img, out_train_lbl),
        ("valid", out_val_img, out_val_lbl),
    ]:
        img_dir = ROBOFLOW_DIR / split / "images"
        lbl_dir = ROBOFLOW_DIR / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / (stem + ".txt")
            if not lbl_path.exists():
                continue
            # remap classes
            lines = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                old_cls = int(float(parts[0]))
                new_cls = ROBOFLOW_MAP.get(old_cls)
                if new_cls is None:
                    continue
                parts[0] = str(new_cls)
                lines.append(" ".join(parts))
            if not lines:
                continue
            # write label
            new_stem = f"rf_{stem}"
            (out_lbl_dir / f"{new_stem}.txt").write_text("\n".join(lines) + "\n")
            # symlink image
            dst = out_img_dir / f"{new_stem}{img_path.suffix}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(img_path.resolve())
            for l in lines:
                counts[int(l.split()[0])] += 1
    return counts


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Building combined shot classification dataset")
    print("=" * 60)

    # --- output dirs ---
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = OUTPUT_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # --- step 1: parse + subsample shot sequences ---
    print("\n[1/5] Parsing shot CSVs...")
    sequences = parse_shot_csvs()
    picks = subsample_sequences(sequences)
    print(f"  {len(sequences)} sequences → {len(picks)} subsampled frames (every {SUBSAMPLE_EVERY}rd)")

    # --- step 2: load pose JSONs ---
    print("\n[2/5] Loading pose annotations...")
    pose_index = load_pose_index()

    # --- step 3: auto-label padelTracker100 frames ---
    print("\n[3/5] Auto-labeling padelTracker100 shots...")
    pt100_labels: dict[str, tuple[int, float, float, float, float]] = {}
    # output_name → (class_id, cx, cy, w, h)
    pt100_counts = Counter()
    skipped = 0

    for prefix, frame_idx, category in picks:
        cls_id = PT100_MAP.get(category)
        if cls_id is None:
            skipped += 1
            continue

        anns = pose_index.get(prefix, {}).get(frame_idx)
        if not anns:
            skipped += 1
            continue

        hitter = identify_hitter(anns)
        if hitter is None:
            skipped += 1
            continue

        bbox = hitter.get("bbox")
        if not bbox or len(bbox) < 4:
            skipped += 1
            continue

        # image dimensions (all frames are 1920×1080)
        cx, cy, w, h = coco_bbox_to_yolo(bbox, 1920, 1080)
        # clamp to [0, 1]
        cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
        w, h = max(0.001, min(1, w)), max(0.001, min(1, h))

        out_name = f"{prefix}_frame_{frame_idx:06d}"
        pt100_labels[out_name] = (cls_id, cx, cy, w, h)
        pt100_counts[cls_id] += 1

    print(f"  labeled: {len(pt100_labels)} frames")
    print(f"  skipped: {skipped} frames (no pose data or unknown category)")
    for cls_id in range(NC):
        print(f"    {CLASS_NAMES[cls_id]}: {pt100_counts[cls_id]}")

    # --- temporal split for padelTracker100 ---
    # Sort by (prefix, frame_idx) then take first 85% per video
    by_video: dict[str, list[str]] = defaultdict(list)
    for name in pt100_labels:
        prefix = name.split("_")[0]  # FinalF or FinalM
        by_video[prefix].append(name)
    for prefix in by_video:
        by_video[prefix].sort()

    train_names: list[str] = []
    val_names: list[str] = []
    for prefix, names in by_video.items():
        split_idx = int(len(names) * TEMPORAL_SPLIT)
        train_names.extend(names[:split_idx])
        val_names.extend(names[split_idx:])
    print(f"  split: {len(train_names)} train / {len(val_names)} val")

    # --- write padelTracker100 labels + symlinks ---
    out_train_img = OUTPUT_DIR / "images/train"
    out_val_img = OUTPUT_DIR / "images/val"
    out_train_lbl = OUTPUT_DIR / "labels/train"
    out_val_lbl = OUTPUT_DIR / "labels/val"

    for name_list, img_dir, lbl_dir in [
        (train_names, out_train_img, out_train_lbl),
        (val_names, out_val_img, out_val_lbl),
    ]:
        for name in name_list:
            cls_id, cx, cy, w, h = pt100_labels[name]
            lbl_dir.joinpath(f"{name}.txt").write_text(
                f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            )
            prefix = name.split("_")[0]
            frame_idx = int(name.split("_")[-1])
            src = PT100_FRAMES / prefix / f"frame_{frame_idx:06d}.jpg"
            if not src.exists():
                continue
            dst = img_dir / f"{name}.jpg"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

    # --- step 4: process Roboflow ---
    print("\n[4/5] Merging Roboflow data (remapped)...")
    rf_counts = process_roboflow(out_train_img, out_val_img, out_train_lbl, out_val_lbl)
    for cls_id in range(NC):
        print(f"    {CLASS_NAMES[cls_id]}: +{rf_counts[cls_id]} from Roboflow")

    # --- step 5: write data.yaml ---
    print("\n[5/5] Writing data.yaml...")
    total_counts = pt100_counts + rf_counts
    yaml_lines = [
        "# Auto-generated by scripts/build_shotclass_combined.py",
        f"# Combined padelTracker100 ({len(pt100_labels)} imgs) + Roboflow remapped",
        "",
        f"path: {OUTPUT_DIR}",
        "train: images/train",
        "val: images/val",
        "test: images/val",
        "",
        f"nc: {NC}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {name}")
    yaml_lines.append("")
    yaml_lines.append("# Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"#   {name}: {total_counts[i]}")
    (OUTPUT_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    # --- summary ---
    total_imgs = len(list(out_train_img.iterdir())) + len(list(out_val_img.iterdir()))
    print("\n" + "=" * 60)
    print(f"DONE: {total_imgs} images, {sum(total_counts.values())} annotations")
    print(f"Output: {OUTPUT_DIR}")
    print("\nFinal class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {total_counts[i]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
