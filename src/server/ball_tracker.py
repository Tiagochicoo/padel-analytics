"""Pure-NumPy ByteTrack + Kalman-filter ball tracker for the Jetson edge server.

This is a self-contained, **OpenCV-free** ball tracker. It consumes the noisy
ball-point detections produced by the Jetson's TrackNet heatmap argmax and
emits stable, Kalman-smoothed ball positions with persistent track IDs. All
math is NumPy/SciPy; the GPU is only used upstream by TrackNet, so this module
runs identically on the Jetson and in CPU-only tests.

==============================================================================
Integration notes — Jetson ``gpu_analyzer.py``
==============================================================================

On the Jetson, ``~/rep/padel-analytics/src/server/gpu_analyzer.py`` currently
detects the ball each frame by taking the ``argmax`` of the TrackNet heatmap.
That yields a single ``(x, y)`` point and a confidence (the heatmap value at
the argmax). There is **no tracking** today, so the reported ball position is
noisy and jumps frame-to-frame.

This module provides drop-in tracking + smoothing. Typical use inside
``GpuAnalyzer._run_gpu_analysis`` (per frame)::

    from edge.inference.ball_tracker import ByteTrackBall

    # once, in __init__ / setup:
    self.ball_tracker = ByteTrackBall()        # tune thresholds via kwargs

    # ---- every frame, after you have the heatmap argmax detection(s) ----
    # detections: an (N, 3) array/list of [x, y, score]; or (N, 2) of [x, y].
    self.ball_tracker.predict(dt=1.0)          # advance the Kalman filters
    tracks = self.ball_tracker.update(detections)   # match + smooth

    if tracks:
        ball = tracks[0]                       # primary (best) active ball
        bx, by   = ball.x, ball.y              # smoothed position (px)
        vx, vy   = ball.vx, ball.vy            # velocity (px/frame)
        track_id = ball.track_id               # stable across frames
        # feed (bx, by, vx, vy, track_id) into rally / bounce analysis

    # During a short gap (no detection this frame) you can still draw the
    # Kalman-predicted ball via the primary track:
    primary = self.ball_tracker.primary_track()   # may be non-None during gaps

Detection input format
----------------------
:meth:`ByteTrackBall.update` accepts any of:
  * an ``(N, 3)`` ndarray / nested list of ``[x, y, score]``
  * an ``(N, 2)`` ndarray / nested list of ``[x, y]``  (score defaults to 1.0)
  * a list of ``(x, y, score)`` or ``(x, y)`` tuples
Scores are in ``[0, 1]``; positions are in image pixels at any resolution.

Why ByteTrack for a single ball?
--------------------------------
ByteTrack's key trick is **two-stage association**: high-confidence detections
are matched first, then *low-confidence* detections are matched against the
still-unmatched tracks. Fast padel balls are frequently blurred and produce
sub-threshold (low-confidence) detections that a naive greedy matcher would
discard; the second stage recovers them. The Kalman filter then smooths the
trajectory and predicts across the inevitable detection gaps.

Dependencies
------------
* ``numpy`` (required) — all state and linear algebra.
* ``scipy`` (optional) — used only for optimal Hungarian assignment via
  ``scipy.optimize.linear_sum_assignment``. If SciPy is unavailable a pure-NumPy
  greedy matcher with identical semantics is used, so the module is fully
  functional with NumPy alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# SciPy is optional: it only provides optimal (Hungarian) assignment. A
# pure-NumPy greedy matcher is used as a fallback so this module has no hard
# dependency beyond NumPy.
try:  # pragma: no cover - import guard
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - SciPy is optional
    linear_sum_assignment = None  # type: ignore[assignment]
    _HAS_SCIPY = False


# ===================================================================== #
#  Kalman filter: 2D constant-velocity model                            #
# ===================================================================== #
class KalmanFilter:
    """2D constant-velocity Kalman filter (pure NumPy).

    State vector ``x`` (4,): ``[px, py, vx, vy]`` — position and velocity.
    Measurement ``z`` (2,): ``[px, py]`` — observed position (px).

    The constant-velocity motion model assumes velocity is perturbed by random
    acceleration (process noise). This is the standard model for smoothly
    moving targets and recovers gracefully after bounces / direction changes,
    because the process-noise covariance lets the filter trust new
    measurements again after a short transient.

    Parameters
    ----------
    init_pos:
        Initial ``(x, y)`` measurement used to seed the state. Velocity starts
        at zero and is learned within a few frames.
    process_accel_std:
        Standard deviation of the (unmodelled) acceleration noise, in
        ``px/frame**2``. Larger → more responsive but noisier; smaller →
        smoother but laggier. ``~1.0–4.0`` works well for padel balls.
    measurement_std:
        Standard deviation of the position measurement noise, in px. Sets the
        diagonal of the measurement-noise covariance ``R``. ``~2–6`` is good
        for TrackNet argmax positions.
    init_velocity_std:
        Prior uncertainty on the initial velocity, in px/frame. Large by
        default so the filter locks onto the true velocity within a few frames.
    init_pos_std:
        Prior uncertainty on the initial position, in px.
    """

    def __init__(
        self,
        init_pos,
        process_accel_std: float = 2.5,
        measurement_std: float = 4.0,
        init_velocity_std: float = 20.0,
        init_pos_std: float = 10.0,
    ) -> None:
        self.x = np.array(
            [float(init_pos[0]), float(init_pos[1]), 0.0, 0.0], dtype=np.float64
        )

        # Prior covariance: reasonably confident about position (seeded from a
        # measurement), quite uncertain about velocity.
        self.P = np.diag(
            [
                init_pos_std ** 2,
                init_pos_std ** 2,
                init_velocity_std ** 2,
                init_velocity_std ** 2,
            ]
        ).astype(np.float64)

        # Measurement matrix: we only observe position.
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        self._accel_std = float(process_accel_std)
        self._meas_var = float(measurement_std) ** 2
        self.R = np.diag([self._meas_var, self._meas_var]).astype(np.float64)

        # Innovation covariance S (populated by predict/update/mahalanobis).
        self.S = self.H @ self.P @ self.H.T + self.R

    # ------------------------------------------------------------------ #
    #  Motion / noise matrices (dt-parameterised)                        #
    # ------------------------------------------------------------------ #
    def _transition(self, dt: float) -> np.ndarray:
        """State-transition matrix F for a constant-velocity step of ``dt``."""
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        """Process-noise covariance Q (discretised white-noise acceleration)."""
        s2 = self._accel_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        return np.array(
            [
                [s2 * dt4 / 4.0, 0.0, s2 * dt3 / 2.0, 0.0],
                [0.0, s2 * dt4 / 4.0, 0.0, s2 * dt3 / 2.0],
                [s2 * dt3 / 2.0, 0.0, s2 * dt2, 0.0],
                [0.0, s2 * dt3 / 2.0, 0.0, s2 * dt2],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    #  Standard Kalman equations                                         #
    # ------------------------------------------------------------------ #
    def predict(self, dt: float = 1.0) -> None:
        """Time-update (prediction) step. Advances state and covariance."""
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        # Refresh innovation covariance so gating reflects the prediction.
        self.S = self.H @ self.P @ self.H.T + self.R

    def update(self, z) -> None:
        """Measurement-update (correction) step with measurement ``z = [x, y]``."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        innovation = z - self.H @ self.x               # y = z - H x
        self.S = self.H @ self.P @ self.H.T + self.R    # innovation covariance
        PHt = self.P @ self.H.T                         # (4, 2)
        # Kalman gain K = P H^T S^{-1}, solved (not inverted) for stability.
        K = np.linalg.solve(self.S.T, PHt.T).T
        self.x = self.x + K @ innovation
        # Joseph-form covariance update: symmetric and numerically stable.
        KH = K @ self.H
        I = np.eye(4, dtype=np.float64)
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ self.R @ K.T

    def mahalanobis(self, z) -> float:
        """Squared Mahalanobis distance of measurement ``z`` under the current
        innovation covariance ``S`` (lower = better fit). Used for gating."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        self.S = self.H @ self.P @ self.H.T + self.R
        innovation = z - self.H @ self.x
        return float(innovation @ np.linalg.solve(self.S, innovation))

    # ------------------------------------------------------------------ #
    #  Accessors                                                         #
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> np.ndarray:
        """Smoothed position ``[x, y]`` (copy)."""
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity ``[vx, vy]`` in px/frame (copy)."""
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        """Speed magnitude in px/frame."""
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def state(self) -> np.ndarray:
        """Full state ``[x, y, vx, vy]`` (copy)."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """State covariance ``P`` (copy)."""
        return self.P.copy()


# ===================================================================== #
#  Data containers                                                      #
# ===================================================================== #
@dataclass(slots=True)
class _Track:
    """Internal mutable track record (one per candidate ball)."""

    kf: KalmanFilter
    track_id: int
    age: int = 0               # total frames since creation
    hits: int = 0              # total matched detections
    time_since_update: int = 0  # frames since last successful match
    confirmed: bool = False    # promoted once hits >= min_confirm_hits
    last_score: float = 0.0    # confidence of the most recent matched detection


@dataclass(slots=True)
class TrackedBall:
    """Public, immutable snapshot of a tracked ball for one frame.

    Attributes
    ----------
    track_id:
        Stable identifier that persists across frames for the same ball.
    x, y:
        Kalman-smoothed ball centre, in image pixels.
    vx, vy:
        Estimated velocity in px/frame.
    confidence:
        Confidence (0–1) of the most recent detection matched to this track.
    age:
        Total number of frames the track has existed.
    hits:
        Total number of detections matched to this track.
    time_since_update:
        Frames since the track was last matched to a detection (0 == updated
        this frame).
    """

    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    confidence: float
    age: int
    hits: int
    time_since_update: int


# ===================================================================== #
#  Association helpers                                                  #
# ===================================================================== #
def _greedy_assign(cost: np.ndarray):
    """Pure-NumPy greedy assignment fallback.

    Pairs rows/columns by ascending finite cost, never reusing a row or column.
    Returns ``(matches, unmatched_rows, unmatched_cols)``. Identical semantics
    to the SciPy path for the small track/detection counts we handle.
    """
    n_rows, n_cols = cost.shape
    candidates = [
        (float(cost[i, j]), i, j)
        for i in range(n_rows)
        for j in range(n_cols)
        if np.isfinite(cost[i, j])
    ]
    candidates.sort()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_rows or j in used_cols:
            continue
        matches.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    unmatched_rows = [i for i in range(n_rows) if i not in used_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in used_cols]
    return matches, unmatched_rows, unmatched_cols


def _associate(
    tracks: list[_Track],
    dets_xy: np.ndarray,
    gate_chi2: float,
    max_dist: float,
):
    """Match detections to tracks.

    The matching cost is the squared Mahalanobis distance of each detection
    under each track's Kalman innovation covariance. Two gates reject bad
    pairings up front:

    * a hard Euclidean distance gate (``max_dist`` px), and
    * a statistical gate (``gate_chi2`` ≈ a chi-square threshold, 2 DOF).

    For ball point-detections this Mahalanobis/distance gating is far more
    appropriate than IoU (which is degenerate for points). Returns
    ``(matches, unmatched_track_idx, unmatched_det_idx)`` where ``matches`` is a
    list of ``(track_idx, det_idx)``.
    """
    n_tracks = len(tracks)
    n_dets = dets_xy.shape[0]
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    cost = np.full((n_tracks, n_dets), np.inf, dtype=np.float64)
    for ti, tr in enumerate(tracks):
        pos = tr.kf.position
        for di in range(n_dets):
            dx = dets_xy[di, 0] - pos[0]
            dy = dets_xy[di, 1] - pos[1]
            if dx * dx + dy * dy > max_dist * max_dist:
                continue  # outside the hard Euclidean gate
            maha = tr.kf.mahalanobis(dets_xy[di])
            if maha <= gate_chi2:
                cost[ti, di] = maha

    if _HAS_SCIPY:
        assert linear_sum_assignment is not None  # for type checkers
        try:
            rows, cols = linear_sum_assignment(cost)
            matches = [
                (int(r), int(c))
                for r, c in zip(rows, cols)
                if np.isfinite(cost[r, c])
            ]
        except ValueError:
            # cost matrix is infeasible (all inf) — no matches possible
            matches = []
    else:
        matches, _, _ = _greedy_assign(cost)

    matched_tracks = {m[0] for m in matches}
    matched_dets = {m[1] for m in matches}
    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


# ===================================================================== #
#  ByteTrack for a single class (the ball)                              #
# ===================================================================== #
class ByteTrackBall:
    """Lightweight ByteTrack adapted for single-class ball tracking.

    Maintains a small pool of tracks (padel usually has exactly one ball in
    play). Implements ByteTrack's two-stage association:

    1. **High-confidence** detections (``score >= high_thresh``) are matched
       first against existing tracks.
    2. Remaining **low-confidence** detections (``low_thresh <= score <
       high_thresh``) are matched against still-unmatched tracks.

    Detections below ``low_thresh`` are discarded. New tracks are only created
    from *unmatched high-confidence* detections (never from low-confidence
    ones), and a track must accumulate ``min_confirm_hits`` matches before it
    is reported, which suppresses one-frame TrackNet false positives.

    Parameters
    ----------
    high_thresh:
        Confidence at/above which a detection is "high confidence" (first
        stage). Default ``0.3``.
    low_thresh:
        Minimum confidence for a detection to participate at all (second
        stage). Default ``0.1``.
    max_time_since_update:
        Frames a track is kept alive without a match before deletion
        (the "track buffer"). Default ``30`` (~1 s at 30 fps).
    min_confirm_hits:
        Matched detections required before a new track is confirmed/reported.
        Default ``3``.
    max_tracks:
        Hard cap on simultaneous tracks. Prevents ID explosion from transient
        false positives. Default ``3``.
    match_gate:
        Squared-Mahalanobis gating threshold (≈ chi-square, 2 DOF).
        ``9.0`` ≈ the 99% confidence region.
    max_match_dist:
        Hard Euclidean match gate in px. Pairings farther than this are never
        matched. Generous by default for fast balls.
    process_accel_std, measurement_std, init_velocity_std, init_pos_std:
        Forwarded to each track's :class:`KalmanFilter`.
    """

    def __init__(
        self,
        high_thresh: float = 0.3,
        low_thresh: float = 0.1,
        max_time_since_update: int = 30,
        min_confirm_hits: int = 3,
        max_tracks: int = 3,
        match_gate: float = 9.0,
        max_match_dist: float = 160.0,
        process_accel_std: float = 2.5,
        measurement_std: float = 4.0,
        init_velocity_std: float = 20.0,
        init_pos_std: float = 10.0,
        first_track_id: int = 1,
    ) -> None:
        if not (0.0 <= low_thresh <= high_thresh <= 1.0):
            raise ValueError(
                "require 0 <= low_thresh <= high_thresh <= 1"
            )
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.max_time_since_update = int(max_time_since_update)
        self.min_confirm_hits = int(min_confirm_hits)
        self.max_tracks = int(max_tracks)
        self.match_gate = float(match_gate)
        self.max_match_dist = float(max_match_dist)

        self._process_accel_std = float(process_accel_std)
        self._measurement_std = float(measurement_std)
        self._init_velocity_std = float(init_velocity_std)
        self._init_pos_std = float(init_pos_std)

        self._tracks: dict[int, _Track] = {}
        self._next_id = int(first_track_id)
        # If True, update() auto-calls predict() first so callers may safely
        # call update() alone. Cleared by predict().
        self._needs_predict = False

    # ------------------------------------------------------------------ #
    #  Prediction                                                        #
    # ------------------------------------------------------------------ #
    def predict(self, dt: float = 1.0) -> None:
        """Advance the Kalman filter of every track by one step.

        Call this once per frame *before* :meth:`update`. (If you forget,
        :meth:`update` will call it for you.)
        """
        for tr in self._tracks.values():
            tr.kf.predict(dt)
            tr.age += 1
            tr.time_since_update += 1
        self._needs_predict = False

    # ------------------------------------------------------------------ #
    #  Update (association + correction)                                 #
    # ------------------------------------------------------------------ #
    def update(self, detections=None, dt: float = 1.0) -> list[TrackedBall]:
        """Match ``detections`` to tracks and correct the Kalman filters.

        Parameters
        ----------
        detections:
            Ball-point detections for this frame: ``(N, 3)`` of ``[x, y,
            score]`` or ``(N, 2)`` of ``[x, y]`` (score defaults to 1.0). Also
            accepts a list of tuples. ``None``/empty means "no detection".
        dt:
            Frame step forwarded to the Kalman predict (only used if
            :meth:`predict` was not called manually).

        Returns
        -------
        list[TrackedBall]
            Confirmed tracks that were updated (matched) this frame, sorted so
            ``[0]`` is the primary ball (most hits, then highest confidence).
            Empty if no confirmed track was matched this frame — use
            :meth:`primary_track` to keep drawing the predicted ball across
            short gaps.
        """
        if self._needs_predict:
            self.predict(dt)

        dets = self._normalize(detections)
        if dets.shape[0] > 0:
            high = dets[dets[:, 2] >= self.high_thresh]
            low = dets[
                (dets[:, 2] >= self.low_thresh)
                & (dets[:, 2] < self.high_thresh)
            ]
        else:
            high = np.zeros((0, 3), dtype=np.float64)
            low = np.zeros((0, 3), dtype=np.float64)

        confirmed_items = [
            (tid, tr) for tid, tr in self._tracks.items() if tr.confirmed
        ]
        tentative_items = [
            (tid, tr) for tid, tr in self._tracks.items() if not tr.confirmed
        ]
        updated_this_frame: set[int] = set()

        # ---- Stage 1: HIGH detections vs CONFIRMED tracks ------------ #
        matches, _, unmatched_high_dets = _associate(
            [tr for _, tr in confirmed_items],
            high[:, :2],
            self.match_gate,
            self.max_match_dist,
        )
        for ti, di in matches:
            tid = confirmed_items[ti][0]
            self._apply_match(tid, high[di])
            updated_this_frame.add(tid)

        # ``unmatched_high_dets`` indexes into ``high``; keep them as a list.
        remaining_high_idx = list(unmatched_high_dets)

        # ---- Stage 2: remaining HIGH detections vs TENTATIVE tracks --- #
        if remaining_high_idx and tentative_items:
            sub_high = high[remaining_high_idx, :2]
            matches2, _, unmatched_sub = _associate(
                [tr for _, tr in tentative_items],
                sub_high,
                self.match_gate,
                self.max_match_dist,
            )
            for ti, sub_di in matches2:
                tid = tentative_items[ti][0]
                di = remaining_high_idx[sub_di]
                self._apply_match(tid, high[di])
                updated_this_frame.add(tid)
            remaining_high_idx = [remaining_high_idx[i] for i in unmatched_sub]

        # Existing tracks not matched in stages 1–2 (pool for stage 4). This
        # set must be captured *before* stage 3 creates brand-new tracks.
        stage4_ids = [tid for tid in self._tracks if tid not in updated_this_frame]

        # ---- Stage 3: spawn NEW tracks from unmatched HIGH detections - #
        for di in remaining_high_idx:
            self._create_track(high[di])

        # ---- Stage 4: LOW detections vs remaining existing tracks ----- #
        if low.shape[0] > 0 and stage4_ids:
            stage4_items = [(tid, self._tracks[tid]) for tid in stage4_ids]
            matches4, _, _ = _associate(
                [tr for _, tr in stage4_items],
                low[:, :2],
                self.match_gate,
                self.max_match_dist,
            )
            for ti, di in matches4:
                self._apply_match(stage4_items[ti][0], low[di])
                updated_this_frame.add(stage4_items[ti][0])
        # Unmatched low detections are discarded (ByteTrack never creates
        # tracks from low-confidence detections).

        # ---- Lifecycle: delete stale tracks --------------------------- #
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].time_since_update > self.max_time_since_update:
                del self._tracks[tid]

        # ---- Emit confirmed tracks updated this frame ----------------- #
        output: list[TrackedBall] = []
        for tr in self._tracks.values():
            if tr.confirmed and tr.time_since_update == 0:
                pos = tr.kf.position
                vel = tr.kf.velocity
                output.append(
                    TrackedBall(
                        track_id=tr.track_id,
                        x=float(pos[0]),
                        y=float(pos[1]),
                        vx=float(vel[0]),
                        vy=float(vel[1]),
                        confidence=tr.last_score,
                        age=tr.age,
                        hits=tr.hits,
                        time_since_update=tr.time_since_update,
                    )
                )
        # Primary ball first: most hits, then highest confidence.
        output.sort(key=lambda b: (b.hits, b.confidence), reverse=True)

        self._needs_predict = True  # next update() should predict first
        return output

    # ------------------------------------------------------------------ #
    #  Queries                                                           #
    # ------------------------------------------------------------------ #
    def primary_track(self) -> Optional[TrackedBall]:
        """Best active (confirmed) track, even if not matched this frame.

        Useful to keep drawing the ball across short detection gaps — the
        returned position is the Kalman prediction. Returns ``None`` when no
        confirmed track exists. Selects the track with the smallest
        ``time_since_update`` (most recently seen), breaking ties by hits.
        """
        candidates = [tr for tr in self._tracks.values() if tr.confirmed]
        if not candidates:
            return None
        best = min(candidates, key=lambda tr: (tr.time_since_update, -tr.hits))
        pos = best.kf.position
        vel = best.kf.velocity
        return TrackedBall(
            track_id=best.track_id,
            x=float(pos[0]),
            y=float(pos[1]),
            vx=float(vel[0]),
            vy=float(vel[1]),
            confidence=best.last_score,
            age=best.age,
            hits=best.hits,
            time_since_update=best.time_since_update,
        )

    @property
    def n_tracks(self) -> int:
        """Total number of live tracks (confirmed + tentative)."""
        return len(self._tracks)

    @property
    def n_confirmed(self) -> int:
        """Number of confirmed tracks."""
        return sum(1 for tr in self._tracks.values() if tr.confirmed)

    @property
    def active_track_ids(self) -> list[int]:
        """IDs of all confirmed tracks."""
        return [tid for tid, tr in self._tracks.items() if tr.confirmed]

    def reset(self, first_track_id: int = 1) -> None:
        """Clear all tracks and state (e.g. between rallies / sessions)."""
        self._tracks.clear()
        self._next_id = int(first_track_id)
        self._needs_predict = False

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #
    def _apply_match(self, track_id: int, det: np.ndarray) -> None:
        """Correct a track's Kalman filter with a matched detection."""
        tr = self._tracks[track_id]
        tr.kf.update(det[:2])
        tr.hits += 1
        tr.time_since_update = 0
        tr.last_score = float(det[2])
        if not tr.confirmed and tr.hits >= self.min_confirm_hits:
            tr.confirmed = True

    def _create_track(self, det: np.ndarray) -> Optional[int]:
        """Create a new tentative track from a high-confidence detection.

        Returns the new track id, or ``None`` if the track cap is reached.
        """
        if len(self._tracks) >= self.max_tracks:
            return None
        track_id = self._next_id
        self._next_id += 1
        kf = KalmanFilter(
            init_pos=det[:2],
            process_accel_std=self._process_accel_std,
            measurement_std=self._measurement_std,
            init_velocity_std=self._init_velocity_std,
            init_pos_std=self._init_pos_std,
        )
        tr = _Track(
            kf=kf,
            track_id=track_id,
            age=0,
            hits=1,            # this detection counts as the first hit
            time_since_update=0,
            confirmed=False,
            last_score=float(det[2]),
        )
        self._tracks[track_id] = tr
        # Promote immediately if confirmation requires a single hit.
        if tr.hits >= self.min_confirm_hits:
            tr.confirmed = True
        return track_id

    @staticmethod
    def _normalize(detections) -> np.ndarray:
        """Coerce assorted detection inputs to an ``(N, 3)`` float array
        ``[x, y, score]`` with scores clipped to ``[0, 1]``."""
        if detections is None:
            return np.zeros((0, 3), dtype=np.float64)
        arr = np.asarray(detections, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.ones(arr.shape[0])])
        elif arr.shape[1] >= 3:
            arr = arr[:, :3].copy()
        else:
            raise ValueError(
                "detections must have 2 or 3 columns (x, y[, score]); got "
                f"shape {arr.shape}"
            )
        arr[:, 2] = np.clip(arr[:, 2], 0.0, 1.0)
        return arr


__all__ = ["KalmanFilter", "ByteTrackBall", "TrackedBall"]


# ===================================================================== #
#  Standalone smoke test:  python3 ball_tracker.py                       #
# ===================================================================== #
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)

    # Synthesize a ball moving in a straight line with noisy detections and a
    # couple of dropped frames, then verify the tracker follows it smoothly.
    n_frames = 60
    truth = np.stack(
        [np.linspace(100, 700, n_frames), np.linspace(200, 260, n_frames)], axis=1
    )
    tracker = ByteTrackBall()
    seen_primary = False
    max_err = float("inf")

    for f in range(n_frames):
        if f in (20, 21, 22):  # 3-frame detection gap
            det = None
        else:
            jitter = rng.normal(0.0, 3.0, size=2)
            score = float(rng.uniform(0.15, 0.6))  # mix of low & high conf
            det = [[truth[f, 0] + jitter[0], truth[f, 1] + jitter[1], score]]

        tracker.predict()
        tracks = tracker.update(det)

        primary = tracks[0] if tracks else tracker.primary_track()
        if primary is not None:
            seen_primary = True
            err = np.hypot(primary.x - truth[f, 0], primary.y - truth[f, 1])
            max_err = min(max_err, err) if np.isinf(max_err) else max(max_err, err)

    print(f"scipy available : {_HAS_SCIPY}")
    print(f"tracker reached : {tracker.n_tracks} track(s), "
          f"{tracker.n_confirmed} confirmed")
    print(f"primary seen    : {seen_primary}")
    print(f"max pos error   : {max_err:.2f} px")
    assert seen_primary, "tracker never produced a confirmed track"
    assert max_err < 25.0, f"tracking error too large: {max_err:.1f}px"
    print("OK — ball tracker smoke test passed")
