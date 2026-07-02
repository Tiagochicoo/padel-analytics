#!/usr/bin/env python3
"""
batch_analyze.py
================
CLI tool for batch pre-analysis of padel videos using the 4-model GPU pipeline.
Saves results as JSON in /mnt/pi-recordings/_analysis/ for instant gallery recall.

Usage:
    python batch_analyze.py /mnt/pi-recordings/highlights_day6_women.mp4 --skip 15
    python batch_analyze.py /mnt/pi-recordings/ --skip 15   # analyze all mp4s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.server.gpu_inference import (
    TRTEngine, TrackNetEngine, MODEL_DEFS, TRACKNET_H, TRACKNET_W, TRACKNET_N_FRAMES,
    parse_output,
)

CACHE_DIR = Path("/mnt/pi-recordings/_analysis")


def get_video_info(video_path: str):
    """Get video duration via ffprobe."""
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30)
    duration = float(r.stdout.strip()) if r.stdout.strip() else 30.0
    total_frames = int(duration * 30)
    return duration, total_frames


def open_video_pipe(video_path: str):
    """Open a continuous ffmpeg pipe for raw video frames."""
    import subprocess
    proc = subprocess.Popen([
        "ffmpeg", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    return proc


def analyze_video(video_path: str, skip: int = 15, verbose: bool = True):
    """Run the 4-model pipeline on a video using continuous ffmpeg pipe."""
    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).name
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch Analysis: {video_name}")
        print(f"{'='*60}")
    
    # Load engines
    engines = {}
    tracknet = None
    for name, cfg in MODEL_DEFS.items():
        if not cfg["path"].exists():
            if verbose:
                print(f"  SKIP {name}: engine not found at {cfg['path']}")
            continue
        if verbose:
            print(f"  Loading {name}...", end=" ", flush=True)
        try:
            if cfg["type"] == "tracknet":
                tracknet = TrackNetEngine(cfg["path"])
                engines[name] = tracknet
            else:
                engines[name] = TRTEngine(cfg["path"])
            if verbose:
                print("OK")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")

    if not engines:
        print("ERROR: no engines could be loaded")
        return None

    # Get video info
    duration, total_frames = get_video_info(video_path)
    if verbose:
        print(f"\n  Duration: {duration:.1f}s, Est. frames: {total_frames}")
        print(f"  Stride: {skip}, Will analyze ~{total_frames // skip} frames")
        print(f"  Starting continuous pipe...")

    stride = skip
    tn_buffer = []
    frame_count = 0
    results = []
    
    # Stats accumulators
    model_det_counts = {}
    model_confidences = {n: [] for n in MODEL_DEFS}
    shot_distribution = {}
    
    start_time = time.time()

    # Open continuous ffmpeg pipe
    frame_bytes = 1280 * 720 * 3
    proc = open_video_pipe(video_path)

    try:
        fidx = 0
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break  # EOF

            if fidx % stride != 0:
                fidx += 1
                continue

            frame = np.frombuffer(raw[:frame_bytes], dtype=np.uint8).reshape(720, 1280, 3)
            frame_dets = {}
            frame_count += 1

            # Regular models
            for name, engine in engines.items():
                cfg = MODEL_DEFS.get(name, {})
                if cfg.get("type") == "tracknet":
                    continue
                output = engine.infer(frame)
                cf = list(cfg["classes"].keys()) if cfg.get("classes") else None
                dets = parse_output(output, cfg.get("type", "detection"), class_filter=cf)
                frame_dets[name] = dets

            # TrackNet
            if tracknet is not None:
                tn_buffer.append(frame)
                if len(tn_buffer) > TRACKNET_N_FRAMES:
                    tn_buffer.pop(0)
                if len(tn_buffer) == TRACKNET_N_FRAMES:
                    tracknet.frame_buffer = []
                    for f in tn_buffer:
                        tracknet.add_frame(f)
                    heatmaps = tracknet.infer()
                    balls = tracknet.parse_ball(heatmaps)
                    frame_dets["ball_tracknet"] = balls

            # Stats
            for name, dets in frame_dets.items():
                model_det_counts[name] = model_det_counts.get(name, 0) + len(dets)
                if dets:
                    model_confidences[name].extend([d.get("confidence", d.get("conf", 0)) for d in dets])

            # Shot distribution
            if "shot_classifier" in frame_dets:
                for d in frame_dets["shot_classifier"]:
                    cls_name = MODEL_DEFS["shot_classifier"]["classes"].get(d["class_id"], "Unknown")
                    shot_distribution[cls_name] = shot_distribution.get(cls_name, 0) + 1

            # Build result
            ts = fidx / 30.0
            frame_result = {
                "frame": fidx,
                "timestamp_s": ts,
                "detections": {},
                "stats": {},
            }
            for name, dets in frame_dets.items():
                serializable_dets = []
                for d in dets:
                    sd = dict(d)
                    # Ensure all values are JSON-serializable
                    for k, v in sd.items():
                        if isinstance(v, (np.floating,)):
                            sd[k] = float(v)
                        elif isinstance(v, (np.integer,)):
                            sd[k] = int(v)
                    serializable_dets.append(sd)
                frame_result["detections"][name] = serializable_dets
                confs = [d.get("confidence", d.get("conf", 0)) for d in dets]
                frame_result["stats"][name] = {
                    "count": len(dets),
                    "avg_conf": round(float(np.mean(confs)), 4) if confs else 0.0,
                }
            results.append(frame_result)

            # Progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                pct = min(100, int(fidx / total_frames * 100)) if total_frames else 0
                print(f"  [{video_name}] {frame_count} frames analyzed ({pct}%) | {fps:.1f} infer fps | {elapsed:.0f}s elapsed", flush=True)

            fidx += 1

    finally:
        proc.terminate()
        import subprocess
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    elapsed = time.time() - start_time
    infer_fps = frame_count / elapsed if elapsed > 0 else 0

    if verbose:
        print(f"\n  Done! {frame_count} frames in {elapsed:.1f}s ({infer_fps:.1f} infer fps)")

    # Build summary
    summary = {}
    for name in MODEL_DEFS:
        total_dets = model_det_counts.get(name, 0)
        confs = model_confidences.get(name, [])
        summary[f"{name}_detections"] = total_dets
        summary[f"avg_{name}_confidence"] = round(float(np.mean(confs)), 4) if confs else 0.0
    summary["shot_distribution"] = shot_distribution
    summary["total_frames_analyzed"] = frame_count
    summary["elapsed_s"] = round(elapsed, 1)
    summary["infer_fps"] = round(infer_fps, 1)

    output = {
        "video": video_name,
        "duration_s": duration,
        "fps": 30,
        "frames_analyzed": frame_count,
        "stride": stride,
        "results": results,
        "summary": summary,
    }

    return output


def save_results(output: dict, verbose: bool = True):
    """Save analysis results to cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    video_name = output["video"]
    safe_name = Path(video_name).stem.replace(" ", "_").replace("/", "_")
    cache_file = CACHE_DIR / f"{safe_name}_analysis.json"
    cache_file.write_text(json.dumps(output, indent=2))
    if verbose:
        size_mb = cache_file.stat().st_size / 1024 / 1024
        print(f"  Saved {len(output['results'])} frame results to {cache_file} ({size_mb:.1f} MB)")
    return cache_file


def main():
    parser = argparse.ArgumentParser(description="Batch pre-analyze padel videos with 4-model GPU pipeline")
    parser.add_argument("path", help="Video file path or directory of mp4s")
    parser.add_argument("--skip", type=int, default=15, help="Frame stride (default: 15)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    video_path = Path(args.path)
    
    if video_path.is_dir():
        # Analyze all mp4s in directory
        videos = sorted(video_path.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found in {video_path}")
            return 1
        print(f"Found {len(videos)} videos to analyze")
        for v in videos:
            # Check if already cached
            safe_name = v.stem.replace(" ", "_").replace("/", "_")
            cache_file = CACHE_DIR / f"{safe_name}_analysis.json"
            if cache_file.exists():
                print(f"  Skipping {v.name} (already cached)")
                continue
            try:
                output = analyze_video(str(v), skip=args.skip, verbose=not args.quiet)
                if output:
                    save_results(output, verbose=not args.quiet)
            except KeyboardInterrupt:
                print("\nInterrupted")
                return 1
            except Exception as e:
                print(f"ERROR analyzing {v.name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Single video
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return 1
        output = analyze_video(str(video_path), skip=args.skip, verbose=not args.quiet)
        if output:
            save_results(output, verbose=not args.quiet)
            # Print summary
            s = output["summary"]
            print(f"\n{'='*60}")
            print(f"Summary for {output['video']}:")
            print(f"{'='*60}")
            print(f"  Frames analyzed: {s['total_frames_analyzed']}")
            print(f"  Duration: {output['duration_s']:.0f}s")
            print(f"  Elapsed: {s['elapsed_s']:.1f}s ({s['infer_fps']:.1f} infer fps)")
            print(f"  Detections:")
            for name in MODEL_DEFS:
                key = f"{name}_detections"
                conf_key = f"avg_{name}_confidence"
                if key in s:
                    print(f"    {name}: {s[key]} detections (avg conf: {s.get(conf_key, 0):.3f})")
            if s.get("shot_distribution"):
                print(f"  Shot distribution:")
                for shot, count in sorted(s["shot_distribution"].items(), key=lambda x: -x[1]):
                    print(f"    {shot}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
