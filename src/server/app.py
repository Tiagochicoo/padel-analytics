"""
src/server/app.py
=================
FastAPI web UI for watching live padel analysis.

Run:
    python -m src.server.app
    # or: uvicorn src.server.app:app --host 0.0.0.0 --port 8000

Env knobs:
    YOUTUBE_URL   default video to analyze (falls back to configs/inference.yaml)
    MODEL_WEIGHTS path to a .pt to prefer over the auto-detected padel model
    INFERENCE_DEVICE '0' (GPU, default) | 'cpu'
    PORT          default 8000

Endpoints:
    GET  /                -> viewer page (templates/index.html)
    GET  /video_feed      -> MJPEG stream of annotated frames
    GET  /stats           -> JSON {fps, players, ids, model, source, ...}
    POST /api/reload      -> hot-swap to the latest trained weights
    POST /api/source      -> {url} analyze a different YouTube/file source
    POST /api/pause       -> toggle play/pause
    POST /api/restart     -> restart the video from the beginning
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.server.pipeline import Pipeline  # noqa: E402
from src.server import training_api       # noqa: E402
from src.server.gpu_analyzer import GpuAnalyzerPlugin  # noqa: E402

# ---- configuration ----------------------------------------------------------
with open(PROJECT_ROOT / "configs" / "inference.yaml") as f:
    CFG = yaml.safe_load(f)

DEFAULT_YT = os.getenv(
    "YOUTUBE_URL",
    "https://www.youtube.com/watch?v=gFl3ADnFRtc",  # user-provided match
)
WEIGHTS = os.getenv("MODEL_WEIGHTS", None)
DEVICE = os.getenv("INFERENCE_DEVICE", "0")  # GPU has headroom while training runs
PORT = int(os.getenv("PORT", "8000"))

# Live camera source precedence (Jetson boots straight into the camera):
#   VIDEO_SOURCE > RTSP_URL > CAMERA_INDEX > YOUTUBE_URL (demo fallback).
# CAMERA_INDEX is the /dev/videoN index (e.g. "0"); RTSP_URL an rtsp://... URL.
_CAMERA_INDEX = os.getenv("CAMERA_INDEX")
_RTSP_URL = os.getenv("RTSP_URL")
if os.getenv("VIDEO_SOURCE"):
    DEFAULT_SOURCE, DEFAULT_YT_URL = os.getenv("VIDEO_SOURCE"), None
elif _RTSP_URL:
    DEFAULT_SOURCE, DEFAULT_YT_URL = _RTSP_URL, None
elif _CAMERA_INDEX is not None and _CAMERA_INDEX != "":
    DEFAULT_SOURCE, DEFAULT_YT_URL = int(_CAMERA_INDEX), None
else:
    DEFAULT_SOURCE, DEFAULT_YT_URL = DEFAULT_YT, DEFAULT_YT

# Single shared pipeline (one video analyzed at a time -> simple & robust).
pipeline = Pipeline(
    source=DEFAULT_SOURCE,
    weights=WEIGHTS,
    youtube_url=DEFAULT_YT_URL,
    conf=CFG.get("conf", 0.35),
    iou=CFG.get("iou", 0.5),
    imgsz=CFG.get("imgsz", 640),
    target_fps=25,
    device=DEVICE,
    half=(DEVICE != "cpu"),
)

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "src" / "server" / "templates"))

app = FastAPI(title="Padel Analytics", version="0.1.0")


# ---- routes -----------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    pipeline.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    pipeline.stop()


# Branding for the client-facing surface (rebrandable via env).
BRAND = {
    "name": os.getenv("BRAND_NAME", "Padel Analytics"),
    "tagline": os.getenv("BRAND_TAGLINE", "Pro court intelligence"),
    "accent": os.getenv("BRAND_ACCENT", "#22d3ee"),     # cyan sport-tech accent
    "accent2": os.getenv("BRAND_ACCENT2", "#6366f1"),   # indigo gradient end
    "logo": os.getenv("BRAND_LOGO", ""),                 # optional logo URL
}


# GPU live analysis plugin (6-model TensorRT pipeline)
gpu_plugin = GpuAnalyzerPlugin(app, templates, PROJECT_ROOT, brand=BRAND, nav_active="gpu_live")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Premium client-facing live view — now serves the GPU gallery."""
    return templates.TemplateResponse(request, "gallery.html", {"brand": BRAND, "nav_active": "gallery"})


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    """Operator console: camera source, calibration, model/reID controls."""
    return templates.TemplateResponse(request, "index.html", {"yt": DEFAULT_YT, "brand": BRAND, "nav_active": "admin"})


@app.get("/live", response_class=HTMLResponse)
def live_alias(request: Request):
    return templates.TemplateResponse(request, "live.html", {"brand": BRAND, "nav_active": "live"})


@app.get("/training", response_class=HTMLResponse)
def training_page(request: Request):
    return templates.TemplateResponse(request, "training.html", {"brand": BRAND, "nav_active": "training"})


@app.get("/api/training")
def training_overview():
    return JSONResponse(training_api.overview())


@app.get("/api/training/runs")
def training_runs():
    return JSONResponse(training_api.list_runs())


@app.get("/api/training/runs/{run_id}")
def training_run(run_id: str):
    r = training_api.get_run(run_id)
    if r is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return JSONResponse(r)


@app.get("/api/training/runs/{run_id}/images")
def training_run_images(run_id: str):
    imgs = training_api.list_run_images(run_id)
    return JSONResponse({"run_id": run_id, "images": imgs})


@app.get("/api/training/runs/{run_id}/image/{name}")
def training_run_image(run_id: str, name: str):
    rd = training_api.run_dir_for(run_id)
    if rd is None:
        return Response(status_code=404)
    # guard against path traversal: resolved path must live inside the run dir
    base = rd.resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return Response(status_code=404)
    if not target.is_file() or target.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        return Response(status_code=404)
    return FileResponse(str(target))


@app.get("/api/training/runs/{run_id}/dataset-samples")
def training_dataset_samples(run_id: str):
    """Return random annotated training-image filenames from the run's dataset."""
    result = training_api.list_dataset_samples(run_id, n=24)
    return JSONResponse(result)


@app.get("/api/training/runs/{run_id}/dataset-sample/{filename:path}")
def training_dataset_sample(run_id: str, filename: str):
    """Serve a single training image with YOLO annotations rendered on top."""
    img_bytes = training_api.render_annotated_sample(run_id, filename)
    if img_bytes is None:
        return Response(status_code=404)
    return Response(content=img_bytes, media_type="image/jpeg")


# ---- video stream ----
def _mjpeg_generator():
    """Yield multipart JPEG frames from the pipeline at a steady cadence."""
    boundary = b"--frame\r\n"
    while True:
        jpg = pipeline.latest_jpeg
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1.0 / 15)  # cap client refresh ~15 fps


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats")
def stats():
    s = pipeline.stats
    return JSONResponse(s)


class SourceBody(BaseModel):
    url: str


@app.post("/api/source")
def set_source(body: SourceBody):
    youtube = body.url if ("youtube.com" in body.url or "youtu.be" in body.url) else None
    pipeline.set_source(body.url, youtube_url=youtube)
    return {"ok": True, "source": body.url}


@app.post("/api/reload")
def reload():
    label = pipeline.reload_model()
    return {"ok": True, "model": label}


@app.post("/api/reid")
def trigger_reid():
    """Trigger offline global Re-ID (re-cluster all tracklets)."""
    return pipeline.compute_reid()


@app.post("/api/pause")
def pause():
    return {"ok": True, "paused": pipeline.toggle_pause()}


@app.post("/api/restart")
def restart():
    pipeline.restart()
    return {"ok": True}


# ---- court calibration -------------------------------------------------------
@app.get("/api/calibration")
def get_calibration():
    cal = pipeline.get_calibration()
    if cal is None:
        return {"calibrated": False}
    return {
        "calibrated": True,
        "source_id": cal.source_id,
        "image_corners": cal.image_corners,
        "frame_width": cal.frame_width,
        "frame_height": cal.frame_height,
    }


@app.get("/api/calibration/frame")
def calibration_frame():
    """A static JPEG of the current frame, to click court corners on."""
    jpg = pipeline.calibration_jpeg
    if jpg is None:
        return Response(status_code=503)
    return Response(jpg, media_type="image/jpeg")


class CalibrateBody(BaseModel):
    corners: list[list[float]]   # 4 [x, y] in REFERENCE_CORNERS order
    frame_width: int
    frame_height: int


@app.post("/api/calibrate")
def calibrate(body: CalibrateBody):
    if len(body.corners) != 4:
        return JSONResponse({"ok": False, "error": "need exactly 4 corners"}, status_code=400)
    cal = pipeline.set_calibration(body.corners, body.frame_width, body.frame_height)
    return {"ok": True, "source_id": cal.source_id, "image_corners": cal.image_corners}


@app.post("/api/calibration/clear")
def clear_calibration():
    from src.utils.calibration import delete_calibration
    delete_calibration(pipeline.current_source_id())
    pipeline.reload_model()
    return {"ok": True}


# ── Match lifecycle (Phase 4) ───────────────────────────────────────────────
@app.post("/api/match/start")
def match_start():
    """Manual 'Start Match' — authoritative; discards warmup/prior auto stats."""
    return pipeline.start_match()


@app.post("/api/match/end")
def match_end():
    return pipeline.end_match()


@app.post("/api/match/reset")
def match_reset():
    """Return to warmup for a new match on the same stream."""
    return pipeline.reset_match()


# ── Phase 6 live analytics ──────────────────────────────────────────────────
@app.get("/api/heatmap/{slot}")
def heatmap(slot: int):
    """Per-player position heatmap PNG (KDE over accumulated LIVE court positions)."""
    from src.heatmap import render_heatmap
    pts = pipeline.positions(slot)
    if not pts:
        return Response(status_code=404)
    # subsample for fast KDE on long matches
    if len(pts) > 3000:
        stride = len(pts) // 3000
        pts = pts[::stride]
    png = render_heatmap(pts, title=f"Player {slot}")
    return Response(png, media_type="image/png")


# ── Post-match reports ──────────────────────────────────────────────────────
@app.post("/api/match/report")
def match_report_generate():
    """Build + save the definitive match report. Returns {ok, match_id, report}."""
    state = pipeline.match_state.status.get("state", "")
    if state not in ("live", "ended"):
        return JSONResponse(
            {"ok": False, "reason": "no active match — start a match first"},
            status_code=400,
        )
    report = pipeline.generate_report()
    match_id = report.get("meta", {}).get("match_id", "")
    return {"ok": True, "match_id": match_id, "report": report}


@app.get("/api/match/report")
def match_report_get():
    """Return the last generated report for the current live session (if any)."""
    r = pipeline.last_report()
    if not r:
        return JSONResponse({"error": "no report yet"}, status_code=404)
    return r


@app.get("/api/match/report/asset/{name}")
def match_report_asset(name: str):
    """Serve an asset from the current live session's last report."""
    from src.match_report import report_match_dir
    mid = (pipeline.last_report() or {}).get("meta", {}).get("match_id")
    if not mid:
        return Response(status_code=404)
    p = report_match_dir(mid) / name
    if not p.exists() or not p.is_file():
        return Response(status_code=404)
    media = "image/png" if name.endswith(".png") else "application/octet-stream"
    return FileResponse(str(p), media_type=media)


# ── Saved reports (multi-match library) ─────────────────────────────────────
@app.get("/api/reports")
def reports_list():
    """List all saved match reports (metadata only), newest first."""
    from src.match_report import list_reports
    return JSONResponse(list_reports())


@app.get("/api/reports/{match_id}")
def reports_get(match_id: str):
    """Full report for a specific saved match."""
    from src.match_report import load_report_by_id
    r = load_report_by_id(match_id)
    if r is None:
        return JSONResponse({"error": "report not found"}, status_code=404)
    return r


@app.get("/api/reports/{match_id}/asset/{name}")
def reports_asset(match_id: str, name: str):
    """Serve a heatmap PNG (or other asset) from a saved match report."""
    from src.match_report import report_match_dir
    base = report_match_dir(match_id).resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return Response(status_code=404)
    if not target.is_file() or target.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        return Response(status_code=404)
    media = "image/png" if name.endswith(".png") else "image/jpeg"
    return FileResponse(str(target), media_type=media)


@app.delete("/api/reports/{match_id}")
def reports_delete(match_id: str):
    """Permanently delete a saved match report."""
    from src.match_report import delete_report
    ok = delete_report(match_id)
    if not ok:
        return JSONResponse({"ok": False, "error": "report not found"}, status_code=404)
    return {"ok": True, "deleted": match_id}


@app.post("/api/reports/{match_id}/archive")
def reports_archive(match_id: str):
    """Archive a report (rename with _archived suffix)."""
    from src.match_report import archive_report
    ok = archive_report(match_id)
    if not ok:
        return JSONResponse({"ok": False, "error": "report not found"}, status_code=404)
    return {"ok": True, "archived": match_id}


@app.get("/report", response_class=HTMLResponse)
def report_page(request: Request):
    """Report library: list of all saved matches."""
    return templates.TemplateResponse(request, "report.html", {"brand": BRAND, "nav_active": "report"})


@app.get("/report/{match_id}", response_class=HTMLResponse)
def report_detail_page(request: Request, match_id: str):
    """Detail view for a specific saved match report."""
    return templates.TemplateResponse(request, "report_detail.html",
                                      {"brand": BRAND, "nav_active": "report", "match_id": match_id})


# ── Playback: batch analysis jobs ───────────────────────────────────────────
from src.server.analyze_job import jobs  # noqa: E402


@app.get("/playback", response_class=HTMLResponse)
def playback_page(request: Request):
    """Playback viewer for offline analysis jobs."""
    return templates.TemplateResponse(request, "playback.html",
                                      {"brand": BRAND, "nav_active": "playback"})


@app.get("/api/jobs")
def jobs_list():
    """List all analysis jobs (newest first)."""
    return JSONResponse(jobs.list_jobs())


class JobBody(BaseModel):
    source: str
    force: bool = False


@app.post("/api/jobs")
def jobs_enqueue(body: JobBody):
    """Create (or reuse) an analysis job for the given source."""
    meta = jobs.enqueue(body.source, device=DEVICE, force=body.force)
    return JSONResponse(meta)


@app.get("/api/jobs/{source_id}")
def jobs_get(source_id: str):
    """Get job metadata (state, progress, etc.)."""
    meta = jobs.get_job(source_id)
    if meta is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(meta)


@app.get("/api/jobs/{source_id}/frame/{frame}")
def jobs_frame(source_id: str, frame: int):
    """Get an annotated frame JPEG from a completed job."""
    jpg = jobs.frame_jpeg(source_id, frame)
    if jpg is None:
        return Response(status_code=404)
    return Response(jpg, media_type="image/jpeg")


@app.get("/api/jobs/{source_id}/frame/{frame}/result")
def jobs_frame_result(source_id: str, frame: int):
    """Get per-frame analysis result JSON."""
    result = jobs.frame_result(source_id, frame)
    if result is None:
        return JSONResponse({"error": "frame result not found"}, status_code=404)
    return JSONResponse(result)


@app.get("/api/jobs/{source_id}/stats")
def jobs_stats(source_id: str):
    """Get cumulative per-player stats for a completed job."""
    return JSONResponse({"players": jobs.cumulative_stats(source_id)})


# ── Gallery: list videos for GPU analysis ────────────────────────────────────
import subprocess
from pathlib import Path

RECORDINGS_DIR = Path("/mnt/pi-recordings")


@app.get("/api/videos")
def videos_list():
    """List all .mp4 videos in /mnt/pi-recordings/ with metadata."""
    if not RECORDINGS_DIR.exists():
        return JSONResponse({"videos": []})
    videos = []
    for f in sorted(RECORDINGS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        size_mb = round(f.stat().st_size / 1024 / 1024, 1)
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(f)],
                capture_output=True, text=True, timeout=15)
            duration_s = round(float(r.stdout.strip())) if r.stdout.strip() else 0
        except Exception:
            duration_s = 0
        videos.append({
            "filename": f.name,
            "path": str(f),
            "size_mb": size_mb,
            "duration_s": duration_s,
        })
    return JSONResponse({"videos": videos})


@app.get("/api/video-file")
def video_file(path: str):
    """Serve a video file for canvas-overlay annotated playback."""
    from fastapi.responses import FileResponse
    p = Path(path)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)
    # Security: only allow files from recordings dir
    try:
        p.resolve().relative_to(RECORDINGS_DIR.resolve())
    except ValueError:
        return JSONResponse({"ok": False, "error": "access denied"}, status_code=403)
    return FileResponse(str(p))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=PORT, reload=False)
