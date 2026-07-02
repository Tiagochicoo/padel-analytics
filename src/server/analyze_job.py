"""
analyze_job.py
==============
Offline per-frame analysis: run the full PadelAnalyzer over every frame of a
video ONCE, persist the annotated frames + per-frame results, then expose them
for instant random-access playback/scrubbing in the web UI.

Output layout (per source), under data/analysis/<source_id>/:

    meta.json          job state + video info + progress
    results.jsonl      one JSON object per frame (1-indexed line == frame n)
                       {"frame":n,"players":[{"slot","box","court_xy"}],"shots":[],"n":k}
    frames/000001.jpg  annotated frame (quality 80)

Reuses the SAME analyzer + drawing code as the live pipeline (modelutil +
render), and the SAME manual court calibration (keyed by source_id), so the
offline preview matches what you see live.

A single JobManager runs at most a few jobs concurrently in daemon threads; the
web layer reads meta.json + results.jsonl so it survives restarts.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from src.utils.camera import open_source
from src.utils.calibration import source_id_for
from src.server.modelutil import build_analyzer, resolve_media

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# guards concurrent writes/reads of in-memory caches
_LOCK = threading.Lock()


def _job_dir(source_id: str) -> Path:
    d = ANALYSIS_DIR / source_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames").mkdir(exist_ok=True)
    return d


def _meta_path(source_id: str) -> Path:
    return _job_dir(source_id) / "meta.json"


def _read_meta(source_id: str) -> Optional[dict]:
    p = _meta_path(source_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_meta(meta: dict) -> None:
    meta["updated_at"] = time.time()
    _meta_path(meta["source_id"]).write_text(json.dumps(meta, indent=2))


def _frame_path(source_id: str, frame: int) -> Path:
    return _job_dir(source_id) / "frames" / f"{frame:06d}.jpg"


def _results_path(source_id: str) -> Path:
    return _job_dir(source_id) / "results.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
class JobManager:
    """Owns running analysis jobs + a small in-memory results cache."""

    def __init__(self):
        self._threads: dict[str, threading.Thread] = {}
        self._results_cache: dict[str, list[dict]] = {}

    # ---- discovery --------------------------------------------------------
    def list_jobs(self) -> list[dict]:
        if not ANALYSIS_DIR.exists():
            return []
        out = []
        for d in sorted(ANALYSIS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = _read_meta(d.name)
            if meta:
                out.append(meta)
        return out

    def get_job(self, source_id: str) -> Optional[dict]:
        return _read_meta(source_id)

    # ---- enqueue ----------------------------------------------------------
    def enqueue(self, source: str, requested_weights: Optional[str] = None,
                device: str = "0", force: bool = False) -> dict:
        source_id = source_id_for(source)
        meta = _read_meta(source_id)
        # already done and not forced -> reuse
        if meta and meta.get("state") == "done" and not force:
            return meta
        if meta and meta.get("state") == "running" and source_id in self._threads \
                and self._threads[source_id].is_alive():
            return meta

        # resolve media up front (so failures surface immediately)
        try:
            video_path = resolve_media(source)
        except Exception as e:
            return {"source_id": source_id, "source": source, "state": "failed",
                    "error": f"media: {e}"}

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        meta = {
            "source_id": source_id,
            "source": source,
            "video_path": video_path,
            "state": "running",
            "total_frames": total,
            "frame": 0,
            "fps": fps,
            "width": width,
            "height": height,
            "model": None,
            "calibrated": False,
            "error": None,
            "started_at": time.time(),
            "updated_at": time.time(),
        }
        # reset results file
        _write_meta(meta)
        _results_path(source_id).write_text("")

        t = threading.Thread(
            target=self._run, args=(source_id, video_path, source,
                                    requested_weights, device),
            daemon=True, name=f"job-{source_id}")
        self._threads[source_id] = t
        t.start()
        return meta

    # ---- frame / result access -------------------------------------------
    def frame_jpeg(self, source_id: str, frame: int) -> Optional[bytes]:
        p = _frame_path(source_id, frame)
        if not p.exists():
            return None
        return p.read_bytes()

    def frame_result(self, source_id: str, frame: int) -> Optional[dict]:
        results = self._load_results(source_id)
        if results is None:
            return None
        idx = frame - 1
        if idx < 0 or idx >= len(results):
            return None
        return results[idx]

    def results_slice(self, source_id: str, frm: int, to: int) -> list[dict]:
        results = self._load_results(source_id)
        if results is None:
            return []
        lo = max(0, frm - 1)
        hi = max(lo, to)
        return results[lo:hi]

    def cumulative_stats(self, source_id: str) -> list[dict]:
        """Per-slot cumulative stats over the whole analyzed video."""
        results = self._load_results(source_id)
        if results is None:
            return []
        slots: dict[int, dict] = {}
        for r in results:
            for p in r.get("players", []):
                slot = int(p.get("slot", 0))
                if slot < 1:
                    continue
                st = slots.setdefault(slot, {
                    "slot": slot, "frames": 0, "shots": 0, "distance": 0.0,
                    "prev": None, "last_court_xy": [], "last_box": []})
                st["frames"] += 1
                cxy = p.get("court_xy")
                if cxy and len(cxy) == 2:
                    st["last_court_xy"] = cxy
                    if st["prev"] is not None:
                        d = math.hypot(cxy[0]-st["prev"][0], cxy[1]-st["prev"][1])
                        if d < 5.0:
                            st["distance"] += d
                    st["prev"] = cxy
                if p.get("box"):
                    st["last_box"] = p["box"]
                # shots are attributed per-frame in Phase 3
        out = []
        for slot in sorted(slots):
            st = slots[slot]
            meta = _read_meta(source_id) or {}
            out.append({
                "slot": slot,
                "frames_seen": st["frames"],
                "time_on_court": round(st["frames"] / max(meta.get("fps", 30), 1), 1),
                "distance_m": round(st["distance"], 1),
                "shots": st["shots"],
                "box": st["last_box"],
                "court_xy": st["last_court_xy"],
            })
        return out

    def _load_results(self, source_id: str) -> Optional[list[dict]]:
        with _LOCK:
            if source_id in self._results_cache:
                return self._results_cache[source_id]
        p = _results_path(source_id)
        if not p.exists():
            return None
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({})
        with _LOCK:
            self._results_cache[source_id] = out
        return out

    def invalidate(self, source_id: str) -> None:
        with _LOCK:
            self._results_cache.pop(source_id, None)

    # ---- worker -----------------------------------------------------------
    def _run(self, source_id: str, video_path: str, source: str,
             requested_weights: Optional[str], device: str) -> None:
        print(f"[job:{source_id}] starting analysis of {video_path}")
        try:
            analyzer, label, calibrated = build_analyzer(source, requested_weights, device)
            meta = _read_meta(source_id) or {}
            meta["model"] = label
            meta["calibrated"] = calibrated
            _write_meta(meta)

            src = open_source(video_path)
            res_file = _results_path(source_id)
            cal = getattr(analyzer, "manual_calibration", None)
            n = 0
            with open(res_file, "w") as rf:
                for frame in src.frames():
                    if frame is None:
                        continue
                    n += 1
                    frame = frame.copy()
                    players, shots = [], []
                    try:
                        result = analyzer.analyze(frame)
                        render_result(frame, result, cal)
                        players = [{
                            "slot": int(p.track_id),
                            "box": [float(v) for v in p.box],
                            "court_xy": list(p.court_xy) if p.court_xy else [],
                        } for p in result.players]
                        shots = []
                    except Exception as e:  # pragma: no cover
                        print(f"[job:{source_id}] frame {n} error: {e}")

                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        _frame_path(source_id, n).write_bytes(buf.tobytes())

                    rf.write(json.dumps({
                        "frame": n, "players": players, "shots": shots,
                        "n": len(players),
                    }) + "\n")
                    if n % 20 == 0:
                        rf.flush()
                        meta = _read_meta(source_id) or {}
                        meta["frame"] = n
                        _write_meta(meta)
            src.release()

            self.invalidate(source_id)
            meta = _read_meta(source_id) or {}
            meta["frame"] = n
            meta["state"] = "done"
            _write_meta(meta)
            print(f"[job:{source_id}] done — {n} frames analyzed")
        except Exception as e:  # pragma: no cover
            meta = _read_meta(source_id) or {}
            meta["state"] = "failed"
            meta["error"] = str(e)
            _write_meta(meta)
            print(f"[job:{source_id}] FAILED: {e}")


# shared singleton
jobs = JobManager()
