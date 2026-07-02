#!/usr/bin/env python3
"""
extract_frames.py
=================
Extract every frame from the padelTracker100 match videos as JPG.

Outputs to ``data/datasets/padeltracker100/frames/<video_stem>/frame_NNNNNN.jpg``
matching the naming used in the COCO pose JSONs (frame_000000, frame_000001, ...).

Uses multiprocessing to saturate CPU.  Each worker grabs a contiguous range
of frame indices so we never re-seek within a process.

Usage:
    python scripts/extract_frames.py            # extract all videos
    python scripts/extract_frames.py --skip     # skip already-extracted videos
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "frames"

# (filename, short_name)
VIDEOS = [
    ("2022_BCN_FinalF_1.mp4", "FinalF"),
    ("2022_BCN_FinalM_1.mp4", "FinalM"),
]

JPEG_QUALITY = 95


def _worker(video_path: str, out_dir: str, start: int, end: int, q: Queue):
    """Extract frames [start, end) from *video_path* into *out_dir*."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        q.put(("error", f"Cannot open {video_path}"))
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fourcc_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    idx = start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame, fourcc_params)
        idx += 1
    cap.release()
    q.put(("done", idx - start))


def extract_video(video_path: Path, out_dir: Path, skip: bool, n_workers: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    if skip and existing >= total:
        print(f"  {out_dir.name}: {existing}/{total} frames already extracted, skipping")
        return

    print(f"  {out_dir.name}: extracting {total} frames with {n_workers} workers...")
    chunk = (total + n_workers - 1) // n_workers
    q: Queue = Queue()
    procs = []
    for w in range(n_workers):
        s = w * chunk
        e = min(s + chunk, total)
        if s >= e:
            break
        p = Process(target=_worker, args=(str(video_path), str(out_dir), s, e, q))
        p.start()
        procs.append(p)

    done = 0
    t0 = time.time()
    for _ in procs:
        status, info = q.get()
        if status == "done":
            done += info
            elapsed = time.time() - t0
            fps = done / elapsed if elapsed > 0 else 0
            print(f"    {done}/{total} frames ({fps:.0f} fps)")
        else:
            print(f"    WORKER ERROR: {info}", file=sys.stderr)

    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f"  {out_dir.name}: {done} frames in {elapsed:.0f}s "
          f"({done/elapsed:.0f} fps avg)")


def main():
    ap = argparse.ArgumentParser(description="Extract padelTracker100 video frames")
    ap.add_argument("--skip", action="store_true", help="skip already-extracted videos")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print("==> Extracting padelTracker100 frames")
    for fname, short in VIDEOS:
        vpath = RAW_DIR / fname
        if not vpath.exists():
            print(f"  WARNING: {vpath} not found, skipping")
            continue
        extract_video(vpath, OUT_DIR / short, args.skip, args.workers)
    print("==> Done.")


if __name__ == "__main__":
    main()
