"""
analyzer.py
===========
Multi-model orchestrator for the full padel analysis pipeline.

Combines four swappable models into one coherent analysis:

    PlayerTracker   YOLOv26 (detect+track+reID)   -> tracked players (+ team)
    CourtModel      YOLOv26-pose (keypoints)      -> court corners -> homography -> 2D map
    BallTracker     TrackNetV3/V4                  -> ball position / trajectory
    PoseModel       YOLOv26-pose (body keypoints) -> per-player pose -> shot classification

Each component is OPTIONAL: if its weights are absent it is skipped, so the
system degrades gracefully and lights up feature-by-feature as you train each
model. The web UI consumes AnalysisResult to render boxes, the 2D court map,
the ball trail and shot stats.

This file defines the architecture + interfaces. The player tracker is live
already (training in progress). Court / ball / pose are stubbed and activate
once their weights exist (see docs/ROADMAP.md for the per-component recipe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.cadence import Cadence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _box_iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@dataclass
class Player:
    track_id: int
    box: tuple[float, float, float, float]   # x1,y1,x2,y2 (px)
    team: Optional[int] = None               # 0/1 once reID clustering is added
    court_xy: Optional[tuple[float, float]] = None  # projected 2D-map position (m)
    keypoints: Optional[np.ndarray] = None   # (17,3) COCO body keypoints [x,y,vis]
    outside_court: bool = False              # True if filtered by court polygon
    outside_court: bool = False              # True when player is outside court polygon


@dataclass
class AnalysisResult:
    players: list[Player] = field(default_factory=list)
    court_keypoints: Optional[np.ndarray] = None   # (N,2) px
    homography: Optional[np.ndarray] = None        # 3x3 px->court(m)
    ball_xy: Optional[tuple[float, float]] = None  # px
    ball_trail: list[tuple[float, float]] = field(default_factory=list)
    shots: list[dict] = field(default_factory=list)
    fps: float = 0.0


class PadelAnalyzer:
    """Orchestrates the four sub-models. Load only what you have."""

    def __init__(
        self,
        player_weights: Optional[str] = None,
        court_weights: Optional[str] = None,
        ball_weights: Optional[str] = None,
        pose_weights: Optional[str] = None,
        shotclass_weights: Optional[str] = None,
        device: str = "0",
        manual_calibration=None,
        player_classes: Optional[list[int]] = None,
        reid_mapping: Optional[dict] = None,
        tracker_cfg: str = "configs/botsort_reid.yaml",
        cadence: Optional[Cadence] = None,
        online_reid=None,
    ):
        self.device = device
        self.cadence = cadence or Cadence()
        self._frame_idx = 0
        self.online_reid = online_reid          # OnlineReIDResolver (live P1..P4) or None
        from src.team_assigner import TeamAssigner
        self.team_assigner = TeamAssigner()
        self.manual_calibration = manual_calibration
        self.player_classes = player_classes
        self.reid_mapping = reid_mapping        # raw_id -> canonical (authoritative)
        self.tracker_cfg = tracker_cfg
        self.player = self._try_load(player_weights or "data/models/player_best.pt")
        self.court = self._try_load(court_weights or "data/models/court_best.pt")
        # ball / pose use external frameworks (TrackNet); load lazily when present.
        self.ball = self._load_ball(ball_weights or "data/models/ball_best.pt")
        self.pose = self._try_load(pose_weights)
        self.shotclass = self._try_load(shotclass_weights or "data/models/shotclass_best.pt")
        self.shot_classifier = self._build_shot_classifier()

        # TRT fast path: use GPU TensorRT engine for player detection when available.
        self.trt_player = None
        self.trt_tracker = None
        self._court_cache: Optional[np.ndarray] = None
        self._court_cache_frame: int = -100
        _pe = PROJECT_ROOT / (player_weights or "data/models/player_best.pt")
        _pe = _pe.with_suffix(".engine")
        if _pe.exists():
            try:
                from src.trt_infer import TRTDetector, SimpleTracker
                self.trt_player = TRTDetector(str(_pe))
                self.trt_tracker = SimpleTracker()
                print("[analyzer] TensorRT GPU detection enabled")
            except Exception as e:
                print(f"[analyzer] TRT load failed ({e}), using CPU .pt")

        # Cache the manual homography so we don't recompute it every frame.
        self._manual_H: Optional[np.ndarray] = None
        if manual_calibration is not None:
            from src.utils.calibration import homography_from_calibration
            self._manual_H = homography_from_calibration(manual_calibration)

        # Court-aware 4-player tracker (inside-court filter + stable slot IDs).
        # Built when we have the player model AND either the court model OR a
        # manual court calibration. Falls back to plain detection otherwise.
        if self.player is not None and (self.court is not None or manual_calibration is not None):
            from src.player_tracker import PlayerTracker
            self.ptracker = PlayerTracker(
                self.player, self.court, manual_calibration=manual_calibration,
                tracker_cfg=tracker_cfg,
                player_classes=player_classes,
                court_every_n=self.cadence.court_every)
        else:
            self.ptracker = None

    # ---- loaders (each fails soft -> returns None) -------------------------
    def _try_load(self, path: str):
        if not path:
            return None
        p = PROJECT_ROOT / path
        # Auto-prefer TensorRT engine over PyTorch weights when available
        # AND torch CUDA works (Ultralytics needs GPU torch for engine I/O).
        engine = p.with_suffix('.engine')
        if engine.exists():
            try:
                import torch
                if torch.cuda.is_available():
                    p = engine
            except ImportError:
                pass
        if not p.exists():
            return None
        try:
            from ultralytics import YOLO
            print(f"[analyzer] loading {p.name}")
            return YOLO(str(p))
        except Exception as e:  # pragma: no cover
            print(f"[analyzer] could not load {p}: {e}")
            return None

    def _load_ball(self, weights: Optional[str]):
        """Load the TrackNetV3 ball tracker if weights exist (see src/ball_tracker.py)."""
        if not weights or not Path(weights).exists():
            return None
        try:
            from src.ball_tracker import TrackNetBallTracker
            return TrackNetBallTracker(weights, device=self.device)
        except Exception as e:  # pragma: no cover
            print(f"[analyzer] could not load ball tracker {weights}: {e}")
            return None

    # ---- main per-frame analysis ------------------------------------------
    def analyze(self, frame, prev_result: Optional[AnalysisResult] = None) -> AnalysisResult:
        res = AnalysisResult()
        self._frame_idx += 1
        idx = self._frame_idx - 1

        # 1) players — TRT GPU fast path OR court-aware Ultralytics --------
        if self.trt_player is not None:
            # GPU TensorRT detection + simple centroid tracker
            dets = self.trt_player.detect(frame)
            tracked_pairs = self.trt_tracker.update(dets)
            # Court keypoints via CPU Ultralytics (cadence-gated every 60 frames)
            if self.court is not None and idx - self._court_cache_frame >= self.cadence.court_every:
                try:
                    cr = self.court(frame, verbose=False, imgsz=640)
                    if cr and cr[0].keypoints is not None:
                        kd = cr[0].keypoints.data
                        if len(kd) > 0:
                            self._court_cache = kd[0][:, :2].cpu().numpy()
                            self._court_cache_frame = idx
                except Exception:
                    pass
            res.court_keypoints = self._court_cache
            if self._manual_H is not None:
                res.homography = self._manual_H
            else:
                res.homography = self._homography(res.court_keypoints)
            mapping = (self.online_reid.update(frame, tracked_pairs, idx)
                       if self.online_reid is not None else self.reid_mapping)
            for raw_id, box in tracked_pairs:
                cx = (box[0] + box[2]) / 2
                cy = box[3]
                court_xy = self._project(res.homography, (cx, cy)) if res.homography is not None else None
                tid = mapping.get(raw_id, raw_id) if mapping else raw_id
                team = self.team_assigner.update(tid, court_xy)
                res.players.append(Player(
                    track_id=tid, box=box, court_xy=court_xy, team=team))
        elif self.ptracker is not None:
            tracked = self.ptracker.track(frame)
            res.court_keypoints = self.ptracker.court_keypoints
            # Manual calibration gives a constant homography; otherwise derive
            # it from the detected court keypoints.
            if self._manual_H is not None:
                res.homography = self._manual_H
            else:
                res.homography = self._homography(res.court_keypoints)
            mapping = self._reid_mapping_for(tracked, frame, idx)
            for p in tracked:
                cx = (p.box[0] + p.box[2]) / 2
                cy = p.box[3]                         # feet -> court contact
                court_xy = self._project(res.homography, (cx, cy)) if res.homography is not None else None
                tid = mapping.get(p.raw_id, p.slot) if mapping else p.slot
                team = self.team_assigner.update(tid, court_xy)
                res.players.append(Player(
                    track_id=tid, box=p.box, court_xy=court_xy, team=team,
                    outside_court=p.outside_court))
        elif self.player is not None:
            # fallback: plain detection+track (no court model / calibration yet)
            kw = dict(persist=True, tracker=self.tracker_cfg, verbose=False,
                      conf=0.3, iou=0.5, imgsz=640)
            if self.player_classes is not None:
                kw["classes"] = self.player_classes
            for r in self.player.track(frame, **kw):
                if r.boxes is not None:
                    for b in r.boxes:
                        raw = int(b.id[0]) if b.id is not None else -1
                        tid = self.reid_mapping.get(raw, raw) if self.reid_mapping else raw
                        res.players.append(Player(tid, tuple(b.xyxy[0].cpu().numpy())))

        # 2) ball position / trail ------------------------------------------
        if self.ball is not None and self.cadence.should_run("ball", idx):
            res.ball_xy = self.ball.predict(frame)       # implement in ball_tracker
            if res.ball_xy is not None:
                trail = (prev_result.ball_trail if prev_result else []) + [res.ball_xy]
                res.ball_trail = trail[-30:]
        elif prev_result is not None:
            res.ball_trail = prev_result.ball_trail      # carry frozen trail on skipped frames

        # 3) shot classification (shotclass + ball proximity; pose sharpens it later)
        if (self.shot_classifier is not None and res.ball_xy is not None
                and self.cadence.should_run("shot", idx)):
            res.shots = self.shot_classifier.classify(frame, res.players, res.ball_xy, frame_idx=idx)

        # 4) body pose — per-player COCO keypoints (every N frames for throughput)
        if self.pose is not None and self.cadence.should_run("pose", idx):
            self._run_pose(frame, res, prev_result)
        elif prev_result is not None:
            # carry forward keypoints on non-pose frames
            prev_kpts = {p.track_id: p.keypoints for p in prev_result.players if p.keypoints is not None}
            for p in res.players:
                if p.track_id in prev_kpts:
                    p.keypoints = prev_kpts[p.track_id]

        return res

    # ---- helpers ----------------------------------------------------------
    def _reid_mapping_for(self, tracked, frame, idx):
        """raw_id -> canonical mapping: live online resolver if present, else the
        static offline mapping (or empty -> fall back to tracker slots)."""
        if self.online_reid is not None:
            raw_tracks = [(p.raw_id, p.box) for p in tracked]
            return self.online_reid.update(frame, raw_tracks, idx)
        return self.reid_mapping

    @staticmethod
    def _homography(kpts: Optional[np.ndarray]):
        if kpts is None or len(kpts) < 4:
            return None
        from src.utils.homography import compute_homography
        H, _ = compute_homography(kpts)
        return H

    @staticmethod
    def _project(H, xy):
        if H is None:
            return None
        from src.utils.homography import project_points
        out = project_points(H, [xy])
        return None if out is None else tuple(out[0])

    def _run_pose(self, frame, res: AnalysisResult, prev_result=None) -> None:
        """Run body-pose model on the frame and match keypoints to tracked players."""
        if not res.players:
            return
        try:
            results = self.pose(frame, verbose=False, imgsz=640)
            if not results or results[0].keypoints is None:
                return
            kpts_data = results[0].keypoints.data      # (N, 17, 3)
            if results[0].boxes is None:
                return
            pose_boxes = results[0].boxes.xyxy.cpu().numpy()
            for player in res.players:
                best_iou, best_kpts = 0.0, None
                for i, pbox in enumerate(pose_boxes):
                    iou = _box_iou(player.box, tuple(pbox))
                    if iou > best_iou:
                        best_iou = iou
                        best_kpts = kpts_data[i].cpu().numpy()
                if best_iou > 0.3:
                    player.keypoints = best_kpts
        except Exception:
            pass  # pose is best-effort — don't break the pipeline

    def _build_shot_classifier(self):
        """Wire the shotclass detector into a ShotClassifier (or None if absent)."""
        if self.shotclass is None:
            return None
        try:
            from src.shot_classifier import ShotClassifier
            return ShotClassifier(self.shotclass, device=self.device)
        except Exception as e:  # pragma: no cover
            print(f"[analyzer] could not init shot classifier: {e}")
            return None

    def _classify_shots(self, frame, players: list[Player]) -> list[dict]:
        """Legacy stub — shot classification now runs via self.shot_classifier (ball proximity)."""
        return []
