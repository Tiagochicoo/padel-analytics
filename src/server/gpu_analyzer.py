"""
gpu_analyzer.py
===============
4-model TensorRT GPU analysis plugin for the FastAPI gallery server.
Adds live MJPEG streaming routes when imported into app.py.
Now uses continuous ffmpeg pipe for 50-100x faster video decoding.

Usage in app.py:
    from src.server.gpu_analyzer import GpuAnalyzerPlugin
    gpu_plugin = GpuAnalyzerPlugin(app, templates, PROJECT_ROOT)
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

# TRT engine classes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.server.ball_tracker import ByteTrackBall
from src.server.player_tracker import PlayerTracker
from src.server.gpu_inference import (
    TRTEngine, TrackNetEngine, MODEL_DEFS, TRACKNET_H, TRACKNET_W, TRACKNET_N_FRAMES,
    parse_output, draw_overlay, draw_ball, draw_status_bar,
)


# Cache directory for pre-computed analysis results
CACHE_DIR = Path("/mnt/pi-recordings/_analysis")


class StartBody(BaseModel):
    video: str
    skip: int = 5
    real_time: bool = False


class GpuAnalyzerPlugin:
    """Plug GPU live inference routes into a FastAPI app."""

    def __init__(self, app, templates, project_root: Path, brand: dict = None, nav_active: str = "gpu_live"):
        self.app = app
        self.templates = templates
        self.project_root = project_root
        self._brand = brand or {}
        self._nav_active = nav_active
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._status = {"state": "idle", "video": "", "frame": 0, "total": 0}
        self._live_stats = {
            "models": {},
            "ball_positions": [],
            "ball_trail": [],
        }
        self._stats_updated = 0.0

        # Batch analysis state
        self._batch_jobs = {}
        self._batch_job_id_counter = 0

        self._register_routes()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _register_routes(self):
        @self.app.get("/gpu_live", response_class=HTMLResponse)
        async def gpu_live_page(request: Request):
            return self.templates.TemplateResponse(request, "gpu_live.html",
                                                    {"brand": self._brand, "nav_active": self._nav_active})

        @self.app.post("/api/gpu/start")
        async def gpu_start(body: StartBody):
            video_path = Path(body.video)
            if not video_path.is_absolute():
                video_path = Path("/mnt/pi-recordings") / body.video
            if not video_path.exists():
                return JSONResponse({"ok": False, "error": f"video not found: {video_path}"})
            
            self.stop()
            self._stop.clear()
            self._running = True
            with self._lock:
                self._status = {"state": "starting", "video": str(video_path),
                                "frame": 0, "total": 0}
            
            self._thread = threading.Thread(
                target=self._run_gpu_analysis,
                args=(str(video_path), body.skip, body.real_time),
                daemon=True,
            )
            self._thread.start()
            return {"ok": True, "video": str(video_path), "skip": body.skip}

        @self.app.post("/api/gpu/stop")
        async def gpu_stop():
            self.stop()
            return {"ok": True}


        @self.app.post("/api/gpu/live/start")
        async def gpu_live_start():
            """Start live camera analysis on USB camera (/dev/video0)."""
            import os as _os
            device = _os.getenv("CAMERA_DEVICE", "/dev/video0")
            self.stop()
            self._stop.clear()
            self._running = True
            with self._lock:
                self._status = {"state": "starting", "video": f"LIVE CAMERA ({device})",
                                "frame": 0, "total": 0, "live": True}
            self._thread = threading.Thread(
                target=self._run_camera_analysis,
                args=(device,),
                daemon=True,
            )
            self._thread.start()
            return {"ok": True, "mode": "live", "device": device}

        @self.app.get("/api/gpu/status")
        async def gpu_status():
            with self._lock:
                return dict(self._status)

        @self.app.get("/api/gpu/stats")
        async def gpu_stats():
            with self._lock:
                return dict(self._live_stats)

        @self.app.get("/gpu_feed")
        async def gpu_feed():
            return StreamingResponse(
                self._mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        # ── Batch analysis endpoints ──
        @self.app.post("/api/gpu/analyze-batch")
        async def gpu_analyze_batch(body: StartBody):
            """Start batch analysis on a video, returns immediately with job_id."""
            video_path = Path(body.video)
            if not video_path.is_absolute():
                video_path = Path("/mnt/pi-recordings") / body.video
            if not video_path.exists():
                return JSONResponse({"ok": False, "error": f"video not found: {video_path}"})
            
            self._batch_job_id_counter += 1
            job_id = f"batch_{self._batch_job_id_counter}"
            
            job_info = {
                "job_id": job_id,
                "video": str(video_path),
                "skip": body.skip,
                "state": "queued",
                "progress": 0,
                "total_frames": 0,
                "frames_analyzed": 0,
                "error": None,
            }
            self._batch_jobs[job_id] = job_info
            
            t = threading.Thread(
                target=self._run_batch_analysis,
                args=(job_id, str(video_path), body.skip),
                daemon=True,
            )
            t.start()
            return {"ok": True, "job_id": job_id}

        @self.app.get("/api/gpu/batch-status/{job_id}")
        async def gpu_batch_status(job_id: str):
            job = self._batch_jobs.get(job_id)
            if job is None:
                return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
            return JSONResponse(dict(job))

        @self.app.get("/api/gpu/results/{video_name:path}")
        async def gpu_results(video_name: str):
            """Returns cached analysis JSON if available."""
            # Sanitize: derive safe filename
            safe_name = Path(video_name).stem.replace(" ", "_").replace("/", "_")
            cache_file = CACHE_DIR / f"{safe_name}_analysis.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    return JSONResponse({"ok": True, "cached": True, "data": data})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": f"corrupt cache: {e}"})
            return JSONResponse({"ok": True, "cached": False})

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._running = False
        with self._lock:
            self._status["state"] = "stopped"

    def _mjpeg_generator(self):
        boundary = b"--frame\r\n"
        while True:
            with self._lock:
                jpg = self._latest_jpeg
            if jpg:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)

    def _open_video_pipe(self, video_path: str):
        """Open a continuous ffmpeg pipe for raw video frames."""
        import subprocess
        proc = subprocess.Popen([
            "ffmpeg", "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-",
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        return proc


    def _open_camera_pipe(self, device="/dev/video0", width=1280, height=720):
        """Open a continuous ffmpeg pipe reading from a USB camera via V4L2."""
        import subprocess
        proc = subprocess.Popen([
            "ffmpeg",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", f"{width}x{height}",
            "-framerate", "30",
            "-i", device,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-",
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        return proc

    def _get_video_info(self, video_path: str):
        """Get video duration via ffprobe."""
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30)
        duration = float(r.stdout.strip()) if r.stdout.strip() else 30.0
        total_frames = int(duration * 30)
        return duration, total_frames

    def _run_gpu_analysis(self, video_path: str, skip: int, real_time: bool):
        print(f"[gpu] Starting GPU analysis (continuous pipe): {video_path}")
        import subprocess
        
        # Load engines
        engines = {}
        tracknet = None
        for name, cfg in MODEL_DEFS.items():
            if not cfg["path"].exists():
                print(f"  SKIP {name}: engine not found")
                continue
            print(f"[gpu] Loading {name}...", end=" ", flush=True)
            try:
                if cfg["type"] == "tracknet":
                    tracknet = TrackNetEngine(cfg["path"])
                    engines[name] = tracknet
                else:
                    engines[name] = TRTEngine(cfg["path"])
                print(f"OK")
            except Exception as e:
                print(f"FAILED: {e}")

        if not engines:
            with self._lock:
                self._status = {"state": "failed", "error": "no engines loaded"}
            return

        # Get video info
        duration, total_frames = self._get_video_info(video_path)

        with self._lock:
            self._status = {"state": "running", "video": video_path,
                            "frame": 0, "total": total_frames}

        stride = 1 if real_time else skip
        tn_buffer = []
        frame_count = 0
        model_timings = {n: [] for n in MODEL_DEFS}
        model_det_counts = {}
        model_confidences = {n: [] for n in MODEL_DEFS}
        ball_trail = []
        bt = ByteTrackBall(max_tracks=2, min_confirm_hits=1)
        pt = PlayerTracker(min_track_frames=20, max_players=4, max_age=900, max_age_unconfirmed=90, min_box_h=25)
        bt_frame_counter = 0

        # Open continuous ffmpeg pipe
        frame_bytes = 1280 * 720 * 3
        proc = self._open_video_pipe(video_path)

        try:
            fidx = 0
            while True:
                if self._stop.is_set():
                    break

                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break  # EOF

                if fidx % stride != 0:
                    fidx += 1
                    continue  # skip frame (already read & discarded from pipe)

                frame = np.frombuffer(raw[:frame_bytes], dtype=np.uint8).reshape(720, 1280, 3)
                frame_dets = {}
                frame_count += 1

                # Regular models on this frame
                for name, engine in engines.items():
                    cfg = MODEL_DEFS.get(name, {})
                    if cfg.get("type") == "tracknet":
                        continue
                    if self._stop.is_set():
                        break
                    t0 = time.perf_counter()
                    output = engine.infer(frame)
                    lat_ms = (time.perf_counter() - t0) * 1000
                    cf = list(cfg["classes"].keys()) if cfg.get("classes") else None
                    dets = parse_output(output, cfg.get("type", "detection"), class_filter=cf)
                    model_timings[name].append(lat_ms)
                    frame_dets[name] = dets

                if self._stop.is_set():
                    break

                # TrackNet buffer
                if tracknet is not None:
                    tn_buffer.append(frame)
                    if len(tn_buffer) > TRACKNET_N_FRAMES:
                        tn_buffer.pop(0)
                    if len(tn_buffer) == TRACKNET_N_FRAMES:
                        tracknet.frame_buffer = []
                        for f in tn_buffer:
                            tracknet.add_frame(f)
                        if self._stop.is_set():
                            break
                        t0 = time.perf_counter()
                        heatmaps = tracknet.infer()
                        lat_ms = (time.perf_counter() - t0) * 1000
                        balls = tracknet.parse_ball(heatmaps)
                        model_timings["ball_tracknet"].append(lat_ms)
                        frame_dets["ball_tracknet"] = balls

                # Track per-model stats
                for name, dets in frame_dets.items():
                    model_det_counts[name] = model_det_counts.get(name, 0) + len(dets)
                    if dets:
                        model_confidences[name].extend([d.get("confidence", d.get("conf", 0)) for d in dets])

                # Ball tracking via ByteTrack+Kalman
                bt_frame_counter += 1
                raw_balls = frame_dets.get("ball_tracknet", [])
                # Convert to (N,3) [x, y, score] for tracker
                # Use pixel coords for tracking, then normalize for display
                ball_dets = []
                for b in raw_balls:
                    # x, y are normalized 0-1 from TrackNet, convert to pixels
                    px = b["x"] * 1280
                    py = b["y"] * 720
                    score = b.get("conf", 0.5)
                    ball_dets.append([px, py, score])
                import numpy as _np
                det_array = _np.array(ball_dets) if ball_dets else None
                bt.predict()
                bt.update(det_array)
                # Get best track (Kalman-smoothed, even across gaps)
                primary = bt.primary_track()
                if primary is not None:
                    ball_trail.append({"x": primary.x / 1280, "y": primary.y / 720})
                    if len(ball_trail) > 50:
                        ball_trail.pop(0)

                # ── Player tracking (ByteTrack-style slot resolver) ──
                pt.predict()
                # Convert detector person detections to (x1,y1,x2,y2,conf) in pixels
                person_dets = []
                for d in frame_dets.get("detector", []):
                    cx, cy, bw, bh = d["bbox"]
                    x1 = int((cx - bw/2) * 1280)
                    y1 = int((cy - bh/2) * 720)
                    x2 = int((cx + bw/2) * 1280)
                    y2 = int((cy + bh/2) * 720)
                    person_dets.append((x1, y1, x2, y2, d.get("confidence", 0.5)))
                player_tracks = pt.update(person_dets)

                # Annotate frame
                pil_img = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_img)
                w, h = pil_img.size

                # Draw ball trail + positions
                if "ball_tracknet" in frame_dets:
                    draw_ball(draw, frame_dets["ball_tracknet"], w, h)

                # Draw ball trail line
                if len(ball_trail) > 1:
                    for i in range(1, len(ball_trail)):
                        alpha = int(50 + (i / len(ball_trail)) * 200)
                        x1 = int(ball_trail[i-1]["x"] * w)
                        y1 = int(ball_trail[i-1]["y"] * h)
                        x2 = int(ball_trail[i]["x"] * w)
                        y2 = int(ball_trail[i]["y"] * h)
                        draw.line([(x1,y1), (x2,y2)], fill=(239, 68, 68), width=2)

                # Draw player boxes with slot labels (P1-P4)
                try:
                    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                except Exception:
                    font_lg = ImageFont.load_default()
                    font_sm = ImageFont.load_default()

                SLOT_COLORS = {1: "#22c55e", 2: "#3b82f6", 3: "#f59e0b", 4: "#ec4899"}
                for pt_info in player_tracks:
                    bx1, by1, bx2, by2 = pt_info["bbox"]
                    slot = pt_info["slot"]
                    color = SLOT_COLORS.get(slot, "#888")
                    if slot > 0 and pt_info["confirmed"]:
                        label = f"P{slot}"
                        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)
                        # Label background
                        draw.rectangle([bx1, by1-20, bx1+40, by1], fill=color)
                        draw.text((bx1+4, by1-18), label, fill="#fff", font=font_lg)
                    else:
                        # Unconfirmed: thin dashed
                        draw.rectangle([bx1, by1, bx2, by2], outline="#666", width=1)

                # Status bar
                avg_ms = {n: float(np.mean(t) if t else 0) for n, t in model_timings.items()}
                draw_status_bar(draw, w, h, frame_count, fidx, total_frames, avg_ms)

                # Downscale for streaming to cut network load
                stream_w = int(os.getenv("STREAM_WIDTH", "960"))
                stream_q = int(os.getenv("STREAM_QUALITY", "65"))
                if w > stream_w:
                    new_h = int(h * stream_w / w)
                    pil_img = pil_img.resize((stream_w, new_h), Image.LANCZOS)
                buf = io.BytesIO()
                pil_img.save(buf, "JPEG", quality=stream_q)
                with self._lock:
                    self._latest_jpeg = buf.getvalue()
                    self._status["frame"] = fidx
                    # Publish live stats
                    models_stats = {}
                    for name in MODEL_DEFS:
                        avg_conf = 0.0
                        if model_confidences.get(name):
                            avg_conf = round(float(np.mean(model_confidences[name][-100:])), 3)
                        avg_lat = 0.0
                        if model_timings.get(name):
                            avg_lat = round(float(np.mean(model_timings[name][-30:])), 1)
                        models_stats[name] = {
                            "detections": model_det_counts.get(name, 0),
                            "avg_confidence": avg_conf,
                            "avg_latency_ms": avg_lat,
                            "count_this_frame": len(frame_dets.get(name, [])),
                        }
                    self._live_stats = {
                        "models": models_stats,
                        "ball_positions": frame_dets.get("ball_tracknet", []),
                        "ball_trail": list(ball_trail[-20:]),
                        "player_tracks": [{"slot": p["slot"], "cx": p["cx"], "cy": p["cy"], "confirmed": p["confirmed"]} for p in player_tracks],
                        "fps": round(1000 / (avg_lat if avg_lat > 0 else 1), 1) if model_timings.get("detector") else 0,
                    }
                    self._stats_updated = time.time()

                if frame_count % 30 == 0:
                    det_ms = avg_ms.get("detector", 0)
                    tn_ms = avg_ms.get("ball_tracknet", 0)
                    print(f"[gpu] frame {fidx}/{total_frames} ({frame_count} analyzed) | det={det_ms:.0f}ms tn={tn_ms:.0f}ms",
                          flush=True)

                fidx += 1

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        with self._lock:
            self._status["state"] = "done"
        print(f"[gpu] Done — {frame_count} frames analyzed from {video_path}")

    def _run_camera_analysis(self, device="/dev/video0"):
        """Run live GPU analysis on USB camera input (infinite stream)."""
        print(f"[gpu-live] Starting LIVE camera analysis: {device}")
        import subprocess

        # Load engines
        engines = {}
        tracknet = None
        for name, cfg in MODEL_DEFS.items():
            if not cfg["path"].exists():
                print(f"  SKIP {name}: engine not found")
                continue
            print(f"[gpu-live] Loading {name}...", end=" ", flush=True)
            try:
                if cfg["type"] == "tracknet":
                    tracknet = TrackNetEngine(cfg["path"])
                    engines[name] = tracknet
                else:
                    engines[name] = TRTEngine(cfg["path"])
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")

        if not engines:
            with self._lock:
                self._status = {"state": "failed", "error": "no engines loaded"}
            return

        with self._lock:
            self._status = {"state": "running", "video": f"LIVE CAMERA ({device})",
                            "frame": 0, "total": 0, "live": True}

        tn_buffer = []
        frame_count = 0
        fidx = 0
        model_timings = {n: [] for n in MODEL_DEFS}
        model_det_counts = {}
        model_confidences = {n: [] for n in MODEL_DEFS}
        ball_trail = []
        bt = ByteTrackBall(max_tracks=2, min_confirm_hits=1)
        pt = PlayerTracker(min_track_frames=20, max_players=4, max_age=900, max_age_unconfirmed=90, min_box_h=25)
        bt_frame_counter = 0

        frame_bytes = 1280 * 720 * 3
        reconnect_attempts = 0
        MAX_RECONNECT = 5

        while not self._stop.is_set():
            # (Re)open camera pipe
            proc = self._open_camera_pipe(device)
            reconnect_attempts += 1
            print(f"[gpu-live] Camera pipe opened (attempt {reconnect_attempts})")

            try:
                while not self._stop.is_set():
                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        print(f"[gpu-live] Camera stream ended (got {len(raw)} bytes)")
                        break  # EOF or disconnect

                    reconnect_attempts = 0  # reset on successful read
                    frame = np.frombuffer(raw[:frame_bytes], dtype=np.uint8).reshape(720, 1280, 3)
                    frame_dets = {}
                    frame_count += 1

                    # Regular models on this frame
                    for name, engine in engines.items():
                        cfg = MODEL_DEFS.get(name, {})
                        if cfg.get("type") == "tracknet":
                            continue
                        if self._stop.is_set():
                            break
                        t0 = time.perf_counter()
                        output = engine.infer(frame)
                        lat_ms = (time.perf_counter() - t0) * 1000
                        cf = list(cfg["classes"].keys()) if cfg.get("classes") else None
                        dets = parse_output(output, cfg.get("type", "detection"), class_filter=cf)
                        model_timings[name].append(lat_ms)
                        frame_dets[name] = dets

                    if self._stop.is_set():
                        break

                    # TrackNet buffer
                    if tracknet is not None:
                        tn_buffer.append(frame)
                        if len(tn_buffer) > TRACKNET_N_FRAMES:
                            tn_buffer.pop(0)
                        if len(tn_buffer) == TRACKNET_N_FRAMES:
                            tracknet.frame_buffer = []
                            for f in tn_buffer:
                                tracknet.add_frame(f)
                            if self._stop.is_set():
                                break
                            t0 = time.perf_counter()
                            heatmaps = tracknet.infer()
                            lat_ms = (time.perf_counter() - t0) * 1000
                            balls = tracknet.parse_ball(heatmaps)
                            model_timings["ball_tracknet"].append(lat_ms)
                            frame_dets["ball_tracknet"] = balls

                    # Track per-model stats
                    for name, dets in frame_dets.items():
                        model_det_counts[name] = model_det_counts.get(name, 0) + len(dets)
                        if dets:
                            model_confidences[name].extend([d.get("confidence", d.get("conf", 0)) for d in dets])

                    # Ball tracking via ByteTrack+Kalman
                    bt_frame_counter += 1
                    raw_balls = frame_dets.get("ball_tracknet", [])
                    ball_dets = []
                    for b in raw_balls:
                        px = b["x"] * 1280
                        py = b["y"] * 720
                        score = b.get("conf", 0.5)
                        ball_dets.append([px, py, score])
                    import numpy as _np
                    det_array = _np.array(ball_dets) if ball_dets else None
                    bt.predict()
                    bt.update(det_array)
                    primary = bt.primary_track()
                    if primary is not None:
                        ball_trail.append({"x": primary.x / 1280, "y": primary.y / 720})
                        if len(ball_trail) > 50:
                            ball_trail.pop(0)

                    # Player tracking
                    pt.predict()
                    person_dets = []
                    for d in frame_dets.get("detector", []):
                        cx, cy, bw, bh = d["bbox"]
                        x1 = int((cx - bw/2) * 1280)
                        y1 = int((cy - bh/2) * 720)
                        x2 = int((cx + bw/2) * 1280)
                        y2 = int((cy + bh/2) * 720)
                        person_dets.append((x1, y1, x2, y2, d.get("confidence", 0.5)))
                    player_tracks = pt.update(person_dets)

                    # Annotate frame
                    pil_img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size

                    if "ball_tracknet" in frame_dets:
                        draw_ball(draw, frame_dets["ball_tracknet"], w, h)
                    if len(ball_trail) > 1:
                        for i in range(1, len(ball_trail)):
                            x1 = int(ball_trail[i-1]["x"] * w)
                            y1 = int(ball_trail[i-1]["y"] * h)
                            x2 = int(ball_trail[i]["x"] * w)
                            y2 = int(ball_trail[i]["y"] * h)
                            draw.line([(x1,y1), (x2,y2)], fill=(239, 68, 68), width=2)

                    try:
                        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                    except Exception:
                        font_lg = ImageFont.load_default()
                        font_sm = ImageFont.load_default()

                    SLOT_COLORS = {1: "#22c55e", 2: "#3b82f6", 3: "#f59e0b", 4: "#ec4899"}
                    for pt_info in player_tracks:
                        bx1, by1, bx2, by2 = pt_info["bbox"]
                        slot = pt_info["slot"]
                        color = SLOT_COLORS.get(slot, "#888")
                        if slot > 0 and pt_info["confirmed"]:
                            label = f"P{slot}"
                            draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)
                            draw.rectangle([bx1, by1-20, bx1+40, by1], fill=color)
                            draw.text((bx1+4, by1-18), label, fill="#fff", font=font_lg)
                        else:
                            draw.rectangle([bx1, by1, bx2, by2], outline="#666", width=1)

                    # Status bar with LIVE indicator
                    avg_ms = {n: float(np.mean(t) if t else 0) for n, t in model_timings.items()}
                    draw_status_bar(draw, w, h, frame_count, fidx, 0, avg_ms, live=True)

                    # Downscale for streaming
                    stream_w = int(os.getenv("STREAM_WIDTH", "960"))
                    stream_q = int(os.getenv("STREAM_QUALITY", "65"))
                    if w > stream_w:
                        new_h = int(h * stream_w / w)
                        pil_img = pil_img.resize((stream_w, new_h), Image.LANCZOS)
                    buf = io.BytesIO()
                    pil_img.save(buf, "JPEG", quality=stream_q)
                    with self._lock:
                        self._latest_jpeg = buf.getvalue()
                        self._status["frame"] = fidx
                        models_stats = {}
                        for name in MODEL_DEFS:
                            avg_conf = 0.0
                            if model_confidences.get(name):
                                avg_conf = round(float(np.mean(model_confidences[name][-100:])), 3)
                            avg_lat = 0.0
                            if model_timings.get(name):
                                avg_lat = round(float(np.mean(model_timings[name][-30:])), 1)
                            models_stats[name] = {
                                "detections": model_det_counts.get(name, 0),
                                "avg_confidence": avg_conf,
                                "avg_latency_ms": avg_lat,
                                "count_this_frame": len(frame_dets.get(name, [])),
                            }
                        self._live_stats = {
                            "models": models_stats,
                            "ball_positions": frame_dets.get("ball_tracknet", []),
                            "ball_trail": list(ball_trail[-20:]),
                            "player_tracks": [{"slot": p["slot"], "cx": p["cx"], "cy": p["cy"], "confirmed": p["confirmed"]} for p in player_tracks],
                            "fps": round(1000 / (avg_lat if avg_lat > 0 else 1), 1) if model_timings.get("detector") else 0,
                        }
                        self._stats_updated = time.time()

                    if frame_count % 30 == 0:
                        det_ms = avg_ms.get("detector", 0)
                        tn_ms = avg_ms.get("ball_tracknet", 0)
                        print(f"[gpu-live] frame {fidx} ({frame_count} analyzed) | det={det_ms:.0f}ms tn={tn_ms:.0f}ms", flush=True)

                    fidx += 1

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

            if self._stop.is_set():
                break
            if reconnect_attempts >= MAX_RECONNECT:
                print(f"[gpu-live] Max reconnect attempts ({MAX_RECONNECT}) reached, giving up")
                break
            print(f"[gpu-live] Reconnecting to camera in 2s... (attempt {reconnect_attempts + 1})")
            time.sleep(2)

        with self._lock:
            self._status["state"] = "done"
        print(f"[gpu-live] Camera analysis stopped — {frame_count} frames analyzed")

    def _run_batch_analysis(self, job_id: str, video_path: str, skip: int):
        """Run full batch analysis and save results as JSON cache."""
        import subprocess
        print(f"[batch] Starting batch analysis job {job_id}: {video_path}")
        
        # Update job state
        self._batch_jobs[job_id]["state"] = "loading_models"
        
        # Load engines
        engines = {}
        tracknet = None
        for name, cfg in MODEL_DEFS.items():
            if not cfg["path"].exists():
                print(f"[batch] SKIP {name}: engine not found")
                continue
            print(f"[batch] Loading {name}...", end=" ", flush=True)
            try:
                if cfg["type"] == "tracknet":
                    tracknet = TrackNetEngine(cfg["path"])
                    engines[name] = tracknet
                else:
                    engines[name] = TRTEngine(cfg["path"])
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")

        if not engines:
            self._batch_jobs[job_id]["state"] = "failed"
            self._batch_jobs[job_id]["error"] = "no engines loaded"
            return

        # Get video info
        duration, total_frames = self._get_video_info(video_path)
        
        self._batch_jobs[job_id]["state"] = "running"
        self._batch_jobs[job_id]["total_frames"] = total_frames

        stride = skip
        tn_buffer = []
        frame_count = 0
        results = []
        
        # Stats accumulators
        model_det_counts = {}
        model_confidences = {n: [] for n in MODEL_DEFS}
        shot_distribution = {}

        # Open continuous ffmpeg pipe
        frame_bytes = 1280 * 720 * 3
        proc = self._open_video_pipe(video_path)

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

                # Regular models on this frame
                for name, engine in engines.items():
                    cfg = MODEL_DEFS.get(name, {})
                    if cfg.get("type") == "tracknet":
                        continue
                    output = engine.infer(frame)
                    cf = list(cfg["classes"].keys()) if cfg.get("classes") else None
                    dets = parse_output(output, cfg.get("type", "detection"), class_filter=cf)
                    frame_dets[name] = dets

                # TrackNet buffer
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

                # Track per-model stats
                for name, dets in frame_dets.items():
                    model_det_counts[name] = model_det_counts.get(name, 0) + len(dets)
                    if dets:
                        model_confidences[name].extend([d.get("confidence", d.get("conf", 0)) for d in dets])

                # Track shot distribution
                if "shot_classifier" in frame_dets:
                    for d in frame_dets["shot_classifier"]:
                        cls_name = MODEL_DEFS["shot_classifier"]["classes"].get(d["class_id"], "Unknown")
                        shot_distribution[cls_name] = shot_distribution.get(cls_name, 0) + 1

                # Build per-frame result
                ts = fidx / 30.0
                frame_result = {
                    "frame": fidx,
                    "timestamp_s": ts,
                    "detections": {},
                    "stats": {},
                }
                for name, dets in frame_dets.items():
                    frame_result["detections"][name] = dets
                    confs = [d.get("confidence", d.get("conf", 0)) for d in dets]
                    frame_result["stats"][name] = {
                        "count": len(dets),
                        "avg_conf": round(float(np.mean(confs)), 4) if confs else 0.0,
                    }
                results.append(frame_result)

                # Progress
                if frame_count % 100 == 0:
                    pct = min(100, int(frame_count * stride / total_frames * 100)) if total_frames else 0
                    self._batch_jobs[job_id]["progress"] = pct
                    self._batch_jobs[job_id]["frames_analyzed"] = frame_count
                    print(f"[batch] {job_id}: {frame_count} frames analyzed ({pct}%)", flush=True)

                fidx += 1

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Build summary
        summary = {}
        for name in MODEL_DEFS:
            total_dets = model_det_counts.get(name, 0)
            confs = model_confidences.get(name, [])
            summary[f"{name}_detections"] = total_dets
            summary[f"avg_{name}_confidence"] = round(float(np.mean(confs)), 4) if confs else 0.0
        summary["shot_distribution"] = shot_distribution
        summary["total_frames_analyzed"] = frame_count

        # Build final output
        video_name = Path(video_path).name
        output = {
            "video": video_name,
            "duration_s": duration,
            "fps": 30,
            "frames_analyzed": frame_count,
            "stride": stride,
            "results": results,
            "summary": summary,
        }

        # Save to cache
        safe_name = Path(video_path).stem.replace(" ", "_").replace("/", "_")
        cache_file = CACHE_DIR / f"{safe_name}_analysis.json"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(output, indent=2))
        print(f"[batch] Saved {len(results)} frame results to {cache_file}")

        # Update job state
        self._batch_jobs[job_id]["state"] = "done"
        self._batch_jobs[job_id]["progress"] = 100
        self._batch_jobs[job_id]["frames_analyzed"] = frame_count
        self._batch_jobs[job_id]["cache_file"] = str(cache_file)
        print(f"[batch] Job {job_id} complete — {frame_count} frames analyzed")
