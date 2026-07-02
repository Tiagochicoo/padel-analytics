"""
src/server/pipeline.py
======================
The inference engine behind the web UI.

It owns a PadelAnalyzer (court-aware 4-slot player tracking + projection to a
2D court map) plus a video source and runs a background thread that:

  1. resolves the media source (local file, YouTube via yt-dlp, camera, RTSP),
  2. reads frames,
  3. runs the multi-model analyzer -> stable player slots 1..4 + court_xy,
  4. accumulates per-player stats (shots, distance, time on court),
  5. draws boxes / slot labels / court outline + HUD,
  6. encodes each annotated frame to JPEG and exposes it to the web layer,
  7. publishes structured stats (per player) for the UI / minimap.

Model selection (automatic):
  * data/models/player_best.pt  (your trained padel model) if present
  * else falls back to yolo11n.pt restricted to the COCO "person" class,
    so the demo works immediately while training is still running.

Court geometry comes from a manual calibration (the user clicks the 4 court
corners) persisted under data/calibrations/. Without a calibration the tracker
still runs but the 2D minimap is unavailable.

The web layer only touches thread-safe properties (latest_jpeg, stats,
calibration_frame) -> no shared mutable state races.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.camera import open_source                        # noqa: E402
from src.utils.visualization import draw_hud                    # noqa: E402
from src.utils.calibration import (                             # noqa: E402
    Calibration, load_for_source, save_calibration,
    source_id_for,
)
from src.server.stats import StatsAccumulator                   # noqa: E402
from src.server.modelutil import build_analyzer, resolve_media  # noqa: E402
from src.server.render import render_result                     # noqa: E402
from src.server.reid_resolver import ReIDRunner                 # noqa: E402
from src.match_state import MatchState                          # noqa: E402
from src.rules_engine import RulesEngine                        # noqa: E402


class Pipeline:
    def __init__(
        self,
        source: str,
        weights: Optional[str] = None,
        youtube_url: Optional[str] = None,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        target_fps: int = 25,
        device: str = "0",
        half: bool = True,
    ):
        self.source = source
        self.requested_weights = weights
        self.youtube_url = youtube_url
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.target_fps = target_fps
        self.device = device
        self.half = half and device != "cpu"
        # Premium client view: skip the burned-in FPS/obj HUD (draw_hud) so the
        # frame is clean boxes/IDs/court only; score/timer are HTML-overlaid.
        self.clean_hud = os.getenv("CLEAN_HUD", "1") != "0"

        # thread-safe outputs
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None   # for calibration grab
        self._stats: dict = {
            "fps": 0.0, "players": 0, "ids": [], "source": str(source),
            "model": "", "running": False, "calibrated": False,
            "player_stats": [], "court_w": 10.0, "court_h": 20.0,
        }

        self.stats_acc = StatsAccumulator(max_slots=4, court_fps=30.0)

        self.paused = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._analyzer = None        # lazily built on the worker thread
        self.reid = ReIDRunner()     # offline global reID (background compute)
        self.match_state = MatchState()
        self._frame_idx = 0          # match-wide frame counter (drives cadence + warmup)
        # Phase 6 live analytics (accumulated during LIVE only)
        self._rules = RulesEngine()
        self._positions: dict[int, list] = {}   # slot -> [(x,y) metres] for heatmaps
        self._score = [0, 0]                      # points per team [team0, team1]
        self._shot_log: list[dict] = []
        self._MAX_POSITIONS = 6000
        self._MAX_SHOTLOG = 50
        self._ball_history: deque = deque(maxlen=50)  # for serve auto-detect
        self._last_result = None

    # ---- public, thread-safe API ------------------------------------------
    @property
    def stats(self) -> dict:
        with self._lock:
            d = dict(self._stats)
        d["player_stats"] = self.stats_acc.snapshot()
        d["reid"] = self.reid.status
        d["match"] = self.match_state.status
        analyzer = self._analyzer
        resolver = getattr(analyzer, "online_reid", None) if analyzer is not None else None
        d["online_reid"] = resolver.status if resolver is not None else None
        assigner = getattr(analyzer, "team_assigner", None) if analyzer is not None else None
        d["teams"] = assigner.status if assigner is not None else None
        with self._lock:
            d["score"] = list(self._score)
            d["shot_log"] = list(self._shot_log)
            d["rallies"] = self._rules.total_rallies
            d["positions_slots"] = sorted(self._positions.keys())
        return d

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    @property
    def calibration_jpeg(self) -> Optional[bytes]:
        """A JPEG of the current frame for the manual-calibration click UI."""
        with self._lock:
            fr = self._latest_frame
        if fr is None:
            return None
        ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return buf.tobytes() if ok else None

    def current_source_id(self) -> str:
        return source_id_for(self.youtube_url or self.source)

    def set_source(self, source: str, youtube_url: Optional[str] = None) -> None:
        """Switch video source at runtime (restarts the loop)."""
        self.stop()
        self.source = source
        self.youtube_url = youtube_url
        self._reset_live()
        self._last_result = None
        self.start()

    def restart(self) -> None:
        self.stop()
        self._reset_live()
        self._last_result = None
        self.start()

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ---- calibration -------------------------------------------------------
    def get_calibration(self) -> Optional[Calibration]:
        return load_for_source(self.youtube_url or self.source)

    def set_calibration(self, corners, frame_width, frame_height) -> Calibration:
        """Create/replace the manual calibration for the current source."""
        cal = Calibration(
            source_id=self.current_source_id(),
            image_corners=[(float(x), float(y)) for x, y in corners],
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )
        save_calibration(cal)
        # reload so the change takes effect immediately
        self.reload_model()
        return cal

    # ---- model loading -----------------------------------------------------
    def reload_model(self) -> str:
        """(Re)build the analyzer (and pick up new weights/calibration)."""
        src = self.youtube_url or self.source
        self._analyzer, label, calibrated = build_analyzer(
            src, self.requested_weights, self.device, online_reid=True)
        with self._lock:
            self._stats["model"] = label
            self._stats["calibrated"] = calibrated
        print(f"[pipeline] analyzer ready: {label} | calibrated={calibrated}")
        return label

    # ---- offline global reID -----------------------------------------------
    def compute_reid(self) -> dict:
        """Trigger background global Re-ID on the current source.

        When it finishes the analyzer is rebuilt so the live stream immediately
        shows stable P1..P4. Requires the source to be a resolved local file.
        """
        if self.reid.is_running():
            return {"ok": False, "reason": "already computing"}
        src = self.youtube_url or self.source
        # use the resolved local file if possible
        local = src
        try:
            local = resolve_media(self.source, youtube_url=self.youtube_url)
        except Exception:
            pass
        weights, _, _ = _resolve_weights_for_reid(self.requested_weights)
        ok = self.reid.start(
            local, weights, device=self.device,
            on_done=lambda: self.reload_model(),
        )
        return {"ok": ok, "status": self.reid.status}

    # ---- match lifecycle (Phase 4) -----------------------------------------
    def start_match(self) -> dict:
        """Manual 'Start Match' button — AUTHORITATIVE.

        Locks the online Re-ID resolver (freeze P1..P4 for the match), resets
        the stat accumulators (discarding warmup / any prior auto-started
        counting), and enters LIVE. Overrides a previous auto-detect start.
        """
        was_live = self.match_state.counting
        ok = self.match_state.start_by_button(self._frame_idx)
        if not ok:
            return {"ok": False, "reason": "match already ended"}
        analyzer = self._analyzer
        if analyzer is not None and getattr(analyzer, "online_reid", None) is not None:
            analyzer.online_reid.lock_for_match()
        self._reset_live()
        print(f"[pipeline] match START (button) @ frame {self._frame_idx} "
              f"(reset_previous={was_live})")
        return {"ok": True, "started_by": "button", "reset_previous": was_live,
                "frame": self._frame_idx}

    def end_match(self) -> dict:
        self.match_state.end(self._frame_idx)
        return {"ok": True, "state": self.match_state.state}

    def reset_match(self) -> dict:
        """Return to warmup (before a new match on the same stream)."""
        self.match_state.reset()
        self._reset_live()
        return {"ok": True, **self.match_state.status}

    # ---- Phase 6 live analytics (heatmaps / score / shot log) --------------
    def _reset_live(self) -> None:
        with self._lock:
            self.stats_acc.reset()
            self._rules = RulesEngine()
            self._positions = {}
            self._score = [0, 0]
            self._shot_log = []
            self._ball_history.clear()
            self._last_result = None

    def _accumulate_live(self, result) -> None:
        # heatmap positions (skip players outside court — stats paused)
        for p in result.players:
            if p.court_xy is None or p.outside_court:
                continue
            lst = self._positions.setdefault(int(p.track_id), [])
            lst.append((float(p.court_xy[0]), float(p.court_xy[1])))
            if len(lst) > self._MAX_POSITIONS:
                del lst[:1000]
        # rules engine -> shots + points
        for ev in self._rules.update(result, self._frame_idx):
            if ev.kind == "shot":
                self._shot_log.append({
                    "slot": ev.slot, "type": ev.shot_type, "frame": ev.frame,
                    "team": self._team_of(ev.slot, result.players),
                })
                if len(self._shot_log) > self._MAX_SHOTLOG:
                    self._shot_log = self._shot_log[-self._MAX_SHOTLOG:]
            elif ev.kind == "rally_end":
                w = getattr(ev, "winner_team", None)
                if w is not None and 0 <= w < 2:
                    self._score[w] += 1

    @staticmethod
    def _team_of(slot: int, players) -> object:
        for p in players:
            if int(p.track_id) == slot:
                return getattr(p, "team", None)
        return None

    def positions(self, slot: int) -> list:
        """Thread-safe snapshot of a slot's accumulated court positions (metres)."""
        with self._lock:
            return list(self._positions.get(int(slot), []))

    # ---- Phase 7 post-match report -----------------------------------------
    def generate_report(self) -> dict:
        """Build + save the definitive match_report.json (+ final heatmaps)."""
        from src.match_report import build_report, save_report
        report = build_report(self)
        _path, match_id = save_report(report, self.current_source_id(), pipeline=self)
        report.setdefault("meta", {})["match_id"] = match_id
        self._last_report = report
        print(f"[pipeline] match report saved: {match_id}")
        return report

    def last_report(self) -> dict:
        from src.match_report import load_report
        return getattr(self, "_last_report", None) or load_report(self.current_source_id()) or {}

    # ---- main loop ---------------------------------------------------------
    def _run(self) -> None:
        from src.analyzer import PadelAnalyzer  # noqa: F401 (warm import)

        source_path = self._ensure_media()
        self.reload_model()

        with self._lock:
            self._stats["source"] = str(source_path)
            self._stats["running"] = True
            self._stats["camera_connected"] = False

        print(f"[pipeline] source={source_path} | device={self.device}")
        self._fps_ema = 0.0
        self._last_frame_time = time.time()

        while not self._stop.is_set():
            try:
                src = open_source(source_path)
            except Exception as e:  # pragma: no cover
                print(f"[pipeline] could not open source: {e}")
                time.sleep(2.0)
                continue

            if src.is_live:
                self._run_live(src)         # threaded capture, freshest-frame
            else:
                self._run_file(src)         # sequential, every frame

            src.release()
            if self._stop.is_set():
                break
            print("[pipeline] source ended -> restarting loop")

        with self._lock:
            self._stats["running"] = False
            self._stats["camera_connected"] = False

    def _run_live(self, src) -> None:
        """Live camera/RTSP: a background reader supplies the freshest frame only,
        so analysis (slower than the 90 fps capture) never accumulates lag."""
        from src.utils.camera import ThreadedCapture
        cap = ThreadedCapture(src)
        cap.start()
        try:
            while not self._stop.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue
                frame = cap.latest(block=True, timeout=1.0)
                with self._lock:
                    self._stats["camera_connected"] = cap.connected
                if frame is None:
                    continue
                self._handle_frame(frame)
        finally:
            cap.stop()

    def _run_file(self, src) -> None:
        """File/YouTube: sequential playback that processes every frame."""
        now = time.time()
        for frame in src.frames():
            if self._stop.is_set():
                break
            if self.paused:
                time.sleep(0.05)
                continue
            if frame is None:
                continue
            self._handle_frame(frame)
            time.sleep(max(0.0, 1.0 / self.target_fps - (time.time() - now)))
            now = time.time()

    def _handle_frame(self, frame) -> None:
        """Analyze one frame, accumulate LIVE stats, encode + publish the JPEG."""
        frame = frame.copy()
        self._frame_idx += 1
        analyzer = self._analyzer
        player_count = 0

        # Bug-fix: if the analyzer was temporarily disabled after errors, check
        # whether the cooldown has elapsed and attempt re-initialisation.
        if analyzer is None:
            cooldown_end = getattr(self, "_analyzer_retry_at", 0)
            if time.time() >= cooldown_end:
                try:
                    self.reload_model()
                    analyzer = self._analyzer
                    self._consecutive_errors = 0
                    print("[pipeline] analyzer re-initialised after cooldown")
                except Exception:
                    pass

        if analyzer is not None:
            try:
                result = analyzer.analyze(frame, prev_result=getattr(self, "_last_result", None))
                self._consecutive_errors = 0
            except Exception as e:  # pragma: no cover
                self._consecutive_errors = getattr(self, "_consecutive_errors", 0) + 1
                if self._consecutive_errors <= 3:
                    print(f"[pipeline] analyze error: {e}")
                elif self._consecutive_errors == 4:
                    # Cooldown-and-retry instead of permanent disable.
                    cooldown = min(30, 5 * self._consecutive_errors)
                    self._analyzer_retry_at = time.time() + cooldown
                    self._analyzer = None
                    print(f"[pipeline] analyzer disabled for {cooldown}s after {self._consecutive_errors} errors; will retry")
                result = None
            if result is not None:
                self._last_result = result
                self._ball_history.append(result.ball_xy)
                cal = getattr(analyzer, "manual_calibration", None)
                player_count = render_result(frame, result, cal)
                # During WARMUP, check for serve auto-detect
                if self.match_state.state == "warmup":
                    from src.match_state import detect_serve_start
                    if detect_serve_start(self._ball_history, self._frame_idx):
                        print(f"[pipeline] serve auto-detected @ frame {self._frame_idx}")
                        self.start_match()
                # only count stats once the match is LIVE (warmup is
                # calibration-only and discarded)
                if self.match_state.counting:
                    active_players = [p for p in result.players if not p.outside_court]
                    self.stats_acc.update(active_players, result.shots)
                    self._accumulate_live(result)
            else:
                self._ball_history.append(None)

        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps_ema = 0.1 * inst + 0.9 * self._fps_ema

        if not self.clean_hud:
            draw_hud(frame, self._fps_ema, player_count)

        # Downscale for streaming (inference ran at full res; only the
        # preview JPEG is shrunk to cut network load by ~3x).
        stream_w = int(os.getenv("STREAM_WIDTH", "960"))
        stream_q = int(os.getenv("STREAM_QUALITY", "70"))
        if frame.shape[1] > stream_w:
            scale = stream_w / frame.shape[1]
            stream_frame = cv2.resize(
                frame, (stream_w, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            stream_frame = frame

        ok, buf = cv2.imencode(
            ".jpg", stream_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), stream_q],
        )
        if ok:
            with self._lock:
                self._latest_frame = frame          # keep full-res for calibration
                self._latest_jpeg = buf.tobytes()
                self._stats.update(fps=round(self._fps_ema, 1), players=player_count)

    # ---- helpers -----------------------------------------------------------
    def _ensure_media(self) -> str:
        """Resolve the source to a playable local path (download via yt-dlp if needed)."""
        return resolve_media(self.source, youtube_url=self.youtube_url)


def _resolve_weights_for_reid(requested_weights):
    """Pick the same player weights the analyzer uses (keeps raw ids aligned)."""
    from src.server.modelutil import resolve_player_weights
    return resolve_player_weights(requested_weights)
