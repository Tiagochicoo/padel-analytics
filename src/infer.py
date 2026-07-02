"""
infer.py
========
Run inference on a video file, webcam, or RTSP stream. Supports:

    * detection model  -> draws boxes + persistent track IDs (reID via BoT-SORT)
    * pose model       -> draws court keypoints + polygon (homography input)

Examples:
    # Detection (tracking) on a video
    python src/infer.py --model detection --weights data/models/player_best.pt \
        --source data/sample_videos/match.mp4

    # Live webcam
    python src/infer.py --model detection --weights data/models/player_best.pt --source 0

    # Court keypoints
    python src/infer.py --model pose --weights data/models/court_best.pt --source 0

    # An RTSP camera
    python src/infer.py --model detection --weights player_best.pt --source rtsp://...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.camera import open_source                       # noqa: E402
from src.utils.visualization import (                          # noqa: E402
    draw_box, draw_track, draw_keypoints, draw_hud, make_writer,
)


def parse_source(raw: str):
    """'0' -> int(0); anything else stays a string path/url."""
    if raw.isdigit():
        return int(raw)
    return raw


def load_cfg() -> dict:
    with open(PROJECT_ROOT / "configs" / "inference.yaml") as f:
        return yaml.safe_load(f)


def run_detection(model, source_obj, cfg, writer, fps_smooth):
    """Tracking loop for the player detection model."""
    track_cfg = cfg.get("track", {})
    do_track = track_cfg.get("enabled", True)
    color = tuple(cfg.get("draw", {}).get("colors", {}).get("player", (0, 255, 0)))
    show_ids = cfg.get("draw", {}).get("track_ids", True)

    stream = (
        model.track(source=source_obj, stream=True, persist=track_cfg.get("persist", True),
                    tracker=track_cfg.get("tracker", "botsort.yaml"),
                    conf=cfg.get("conf", 0.35), iou=cfg.get("iou", 0.5),
                    imgsz=cfg.get("imgsz", 640), verbose=False)
        if do_track else
        model.predict(source=source_obj, stream=True,
                      conf=cfg.get("conf", 0.35), iou=cfg.get("iou", 0.5),
                      imgsz=cfg.get("imgsz", 640), verbose=False)
    )

    for result in stream:
        frame = result.plot() if not cfg.get("draw", {}).get("boxes", True) else result.orig_img
        n = 0
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                tid = int(box.id[0]) if (do_track and box.id is not None) else None
                if tid is not None and show_ids:
                    draw_track(frame, xyxy, tid, color=color)
                else:
                    draw_box(frame, xyxy, None, color=color)
                n += 1

        fps_smooth = _overlay_and_write(frame, n, fps_smooth, cfg, writer)
        yield frame


def run_pose(model, source_obj, cfg, writer, fps_smooth):
    """Keypoint loop for the court pose model."""
    color = tuple(cfg.get("draw", {}).get("colors", {}).get("court", (0, 165, 255)))
    conf_thresh = cfg.get("conf", 0.35)

    stream = model.predict(source=source_obj, stream=True,
                           conf=conf_thresh, iou=cfg.get("iou", 0.5),
                           imgsz=cfg.get("imgsz", 640), verbose=False)

    for result in stream:
        frame = result.orig_img
        n = 0
        if result.keypoints is not None and len(result.keypoints):
            for kp in result.keypoints:
                # kp.xy: (N,2), kp.conf: (N,1) -> build (N,3)
                xy = kp.xy[0].cpu().numpy()
                if kp.conf is not None:
                    c = kp.conf[0].cpu().numpy().reshape(-1, 1)
                    kpts3 = __import__("numpy").hstack([xy, c])
                else:
                    kpts3 = __import__("numpy").hstack([xy, __import__("numpy").ones((len(xy), 1))])
                draw_keypoints(frame, kpts3, conf=conf_thresh, color=color,
                               draw_skeleton=cfg.get("draw", {}).get("court_polygon", True))
                n += 1

        fps_smooth = _overlay_and_write(frame, n, fps_smooth, cfg, writer)
        yield frame


def _overlay_and_write(frame, n_objects, fps_smooth, cfg, writer):
    """Update FPS (EMA), draw HUD, write to disk if enabled. Returns new fps."""
    now = time.time()
    if not hasattr(_overlay_and_write, "_last"):
        _overlay_and_write._last = now
        _overlay_and_write._fps = 0.0
    dt = now - _overlay_and_write._last
    _overlay_and_write._last = now
    instant = 1.0 / dt if dt > 0 else 0.0
    _overlay_and_write._fps = 0.1 * instant + 0.9 * (_overlay_and_write._fps or instant)

    draw_hud(frame, _overlay_and_write._fps, n_objects)
    if writer is not None and cfg.get("save", False):
        writer.write(frame)
    return _overlay_and_write._fps


def main() -> None:
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description="Padel Analytics inference.")
    p.add_argument("--model", required=True, choices=["detection", "pose"])
    p.add_argument("--weights", required=True, help=".pt or .engine file.")
    p.add_argument("--source", default=None, help="file | cam index | rtsp url")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--save", action="store_true", help="write annotated mp4")
    p.add_argument("--no-show", action="store_true", help="disable GUI window (headless)")
    args = p.parse_args()

    cfg = load_cfg()
    if args.conf is not None:
        cfg["conf"] = args.conf
    cfg["save"] = args.save

    src_raw = args.source or cfg.get("source", "0")
    source = parse_source(str(src_raw))

    source_obj = open_source(source)

    writer = None
    if args.save:
        out_path = PROJECT_ROOT / "data" / "sample_videos" / f"out_{args.model}.mp4"
        writer = make_writer(str(out_path), source_obj.fps or 25,
                             (source_obj.width, source_obj.height))

    model = YOLO(args.weights)
    runner = run_detection if args.model == "detection" else run_pose

    try:
        for _frame in runner(model, source_obj, cfg, writer, 0.0):
            if not args.no_show:
                cv2.imshow("padel-analytics", _frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[infer] interrupted by user")
    finally:
        source_obj.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
