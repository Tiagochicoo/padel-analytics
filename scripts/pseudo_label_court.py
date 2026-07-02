#!/usr/bin/env python3
"""
pseudo_label_court.py
=====================
Run the trained court keypoint model (court_best.pt) on unannotated Plaimaker
images to auto-generate 26-point YOLO-pose labels, then merge with the
existing v1 dataset into a combined court_keypoints_v2 directory.

This 5.5× the dataset size (788 → ~4,400 images) and adds camera-angle diversity.

Usage:
    python scripts/pseudo_label_court.py

Output: data/datasets/court_keypoints_v2/
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COURT_MODEL = PROJECT_ROOT / "data/models/court_best.pt"
PLAIMAKER_DIR = PROJECT_ROOT / "data/datasets/court_plaimaker"
V1_DIR = PROJECT_ROOT / "data/datasets/court_keypoints_26"
OUTPUT_DIR = PROJECT_ROOT / "data/datasets/court_keypoints_v2"

# Confidence threshold for accepting pseudo-labels
CONF_THRESH = 0.5


def collect_plaimaker_images() -> list[Path]:
    """Collect all image paths from Plaimaker train/valid/test."""
    imgs = []
    for split in ("train", "valid", "test"):
        img_dir = PLAIMAKER_DIR / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(p)
    return imgs


def run_pseudo_labeling() -> tuple[int, int]:
    """Run court model on Plaimaker images, write YOLO-pose labels.

    Returns (labeled_count, skipped_count).
    """
    from ultralytics import YOLO

    print(f"[pseudo] loading court model: {COURT_MODEL}")
    model = YOLO(str(COURT_MODEL))
    images = collect_plaimaker_images()
    print(f"[pseudo] {len(images)} Plaimaker images to label")

    # Output dirs for pseudo-labeled data
    pseudo_img_dir = OUTPUT_DIR / "images" / "pseudo"
    pseudo_lbl_dir = OUTPUT_DIR / "labels" / "pseudo"
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    labeled = 0
    skipped = 0

    for i, img_path in enumerate(images):
        results = model.predict(str(img_path), conf=CONF_THRESH, verbose=False)
        r = results[0]

        if r.keypoints is None or len(r.boxes) == 0:
            skipped += 1
            continue

        # Take the highest-confidence detection
        best_idx = int(r.boxes.conf.argmax())
        kp = r.keypoints[best_idx]  # (1, 26, 3) — x, y, conf

        if kp is None or kp.shape[0] == 0:
            skipped += 1
            continue

        # Get bounding box (normalized)
        box = r.boxes[best_idx]
        cx, cy, bw, bh = box.xywhn[0].tolist()

        # Build YOLO-pose label line
        kpt_flat = []
        for k in range(kp.shape[0]):
            kx, ky = float(kp[k, 0]), float(kp[k, 1])
            kv = float(kp[k, 2]) if kp.shape[1] > 2 else 2.0
            # Normalize keypoints (Ultralytics returns pixel coords in pred)
            kpt_flat.extend([kx, ky, kv])

        label_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(
            f"{v:.6f}" for v in kpt_flat
        )

        # Write label
        out_name = f"plaimaker_{img_path.stem}"
        (pseudo_lbl_dir / f"{out_name}.txt").write_text(label_line + "\n")

        # Copy/symlink image
        out_img = pseudo_img_dir / f"{out_name}{img_path.suffix}"
        if not out_img.exists():
            out_img.symlink_to(img_path.resolve())

        labeled += 1
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(images)} ({labeled} labeled, {skipped} skipped)")

    print(f"[pseudo] done: {labeled} labeled, {skipped} skipped")
    return labeled, skipped


def merge_datasets(labeled_count: int) -> None:
    """Merge v1 annotations + pseudo-labels into output directory."""
    print("\n[merge] building combined dataset...")

    # Copy v1 train/val
    for split in ("train", "val"):
        src_img = V1_DIR / "images" / split
        src_lbl = V1_DIR / "labels" / split
        dst_img = OUTPUT_DIR / "images" / split
        dst_lbl = OUTPUT_DIR / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        if src_img.is_dir():
            for p in src_img.iterdir():
                dst = dst_img / p.name
                if not dst.exists():
                    dst.symlink_to(p.resolve())
        if src_lbl.is_dir():
            for p in src_lbl.iterdir():
                dst = dst_lbl / p.name
                if not dst.exists():
                    dst.symlink_to(p.resolve())

    # Split pseudo-labeled images: 90% train, 10% val
    pseudo_imgs = sorted((OUTPUT_DIR / "images" / "pseudo").iterdir())
    split_idx = int(len(pseudo_imgs) * 0.9)
    train_pseudo = pseudo_imgs[:split_idx]
    val_pseudo = pseudo_imgs[split_idx:]

    for img_list, split_name in [(train_pseudo, "train"), (val_pseudo, "val")]:
        dst_img = OUTPUT_DIR / "images" / split_name
        dst_lbl = OUTPUT_DIR / "labels" / split_name
        for img in img_list:
            # Move symlink to split dir
            target = dst_img / img.name
            if not target.exists():
                target.symlink_to(img.resolve())
            lbl = OUTPUT_DIR / "labels" / "pseudo" / (img.stem + ".txt")
            if lbl.exists():
                target_lbl = dst_lbl / lbl.name
                if not target_lbl.exists():
                    target_lbl.symlink_to(lbl.resolve())

    # Count final
    train_count = len(list((OUTPUT_DIR / "images" / "train").iterdir()))
    val_count = len(list((OUTPUT_DIR / "images" / "val").iterdir()))
    print(f"[merge] final: {train_count} train / {val_count} val ({train_count + val_count} total)")


def write_data_yaml() -> None:
    """Write data.yaml for the v2 dataset."""
    # Read v1 config for kpt_shape and flip_idx
    v1_cfg_path = PROJECT_ROOT / "configs" / "pose.yaml"
    with open(v1_cfg_path) as f:
        v1_cfg = yaml.safe_load(f)

    yaml_content = {
        "path": str(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "nc": 1,
        "names": {0: "court"},
        "kpt_shape": v1_cfg.get("kpt_shape", [26, 3]),
        "flip_idx": v1_cfg.get("flip_idx"),
        "flipud": False,
        "fliplr": False,
        "degrees": 0.0,
    }
    (OUTPUT_DIR / "data.yaml").write_text(
        "# Auto-generated by scripts/pseudo_label_court.py\n"
        f"# Court v2: {788} v1 + pseudo-labeled Plaimaker images\n\n"
        + yaml.dump(yaml_content, default_flow_style=False)
    )
    print(f"[yaml] wrote {OUTPUT_DIR / 'data.yaml'}")


def main():
    print("=" * 60)
    print("Court v2 — Pseudo-labeling Plaimaker images")
    print("=" * 60)

    if not COURT_MODEL.exists():
        print(f"ERROR: court model not found at {COURT_MODEL}")
        return

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Step 1: Pseudo-label
    labeled, skipped = run_pseudo_labeling()

    # Step 2: Merge with v1
    merge_datasets(labeled)

    # Step 3: Write data.yaml
    write_data_yaml()

    print("\n" + "=" * 60)
    print(f"DONE. Output: {OUTPUT_DIR}")
    print(f"  Pseudo-labeled: {labeled} images")
    print(f"  Skipped (no detection): {skipped}")
    print(f"  Total with v1: {labeled + 788} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
