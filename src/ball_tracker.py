"""
src/ball_tracker.py
===================
TrackNetV3 ball tracker — the inference module consumed by ``PadelAnalyzer``
in place of the old YOLO ball stub (``src/analyzer.py::_load_ball``).

TrackNet regresses a ball-centre heatmap per frame from a stack of ``seq_len``
consecutive RGB frames plus a median background (bg_mode='concat' -> 27-channel
input). For streaming inference we therefore keep a rolling window of the last
``seq_len`` frames, estimate a median background from the first frames we see,
run one forward pass per new frame and decode the LAST predicted heatmap to a
ball centre in original-image pixels. A constant-velocity Kalman filter
stabilises the centre into a smooth trail and bridges brief detection misses.

Weight-compatible with the vendored ``TrackNet`` (third_party/tracknetv3/) and
with checkpoints produced by ``src/train_tracknet.py`` (their ``param_dict``
carries seq_len/bg_mode, so the model is rebuilt to match before loading).

Interface (matches ``analyzer.py``):
    tracker = TrackNetBallTracker(weights, device=...)
    xy = tracker.predict(frame_bgr)        # -> (x, y) in original px, or None
    trail = tracker.trail                   # list of recent (x, y)
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.tracknet import HEIGHT, WIDTH, get_model


class _Kalman2D:
    """Constant-velocity Kalman filter on (x, y) with per-axis [pos, vel] state."""

    def __init__(self, process_noise: float = 4.0, meas_noise: float = 4.0):
        dt = 1.0
        self.F = np.array([[1, dt], [0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0]], dtype=np.float32)
        self.Q = np.eye(2, dtype=np.float32) * process_noise
        self.R = np.array([[meas_noise]], dtype=np.float32)
        self.x = np.zeros((2, 1), dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32)
        self._init = False

    def reset(self) -> None:
        self._init = False
        self.x[:] = 0
        self.P[:] = np.eye(2)

    def update(self, meas: Optional[float]):
        if meas is None:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return float(self.x[0, 0]) if self._init else None
        if not self._init:
            self.x[0, 0] = meas
            self._init = True
            return float(meas)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = np.array([[meas]], dtype=np.float32) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0])


class TrackNetBallTracker:
    def __init__(
        self,
        weights: str | Path,
        device: str = "0",
        seq_len: Optional[int] = None,
        bg_mode: str = "concat",
        threshold: float = 0.5,
        median_frames: int = 30,
        trail_len: int = 30,
    ):
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu")
        self.threshold = threshold
        self.median_frames = median_frames
        self.trail_len = trail_len

        ckpt = torch.load(weights, map_location="cpu")
        param_dict = ckpt.get("param_dict", {})
        self.seq_len = int(seq_len or param_dict.get("seq_len", 8))
        self.bg_mode = param_dict.get("bg_mode", bg_mode)
        self.model = get_model("TrackNet", self.seq_len, self.bg_mode)
        ms, un = self.model.load_state_dict(ckpt["model"], strict=False)
        if ms or un:
            raise RuntimeError(f"TrackNet weight mismatch loading {weights}: missing={ms} unexpected={un}")
        self.model.to(self.device).eval()

        self._buf: collections.deque = collections.deque(maxlen=self.seq_len)
        self._median_accum: list[np.ndarray] = []
        self._median: Optional[np.ndarray] = None
        self._orig_hw: Optional[tuple[int, int]] = None
        self._kx = _Kalman2D()
        self._ky = _Kalman2D()
        self.trail: list[tuple[float, float]] = []
        self._misses = 0

    def _ensure_median(self, rgb_small: np.ndarray) -> None:
        if self._median is not None or len(self._median_accum) >= self.median_frames:
            if self._median is None and self._median_accum:
                self._median = np.median(np.stack(self._median_accum), axis=0).astype(np.uint8)
            return
        self._median_accum.append(rgb_small)
        if len(self._median_accum) == self.median_frames:
            self._median = np.median(np.stack(self._median_accum), axis=0).astype(np.uint8)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> Optional[tuple[float, float]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._orig_hw is None:
            self._orig_hw = rgb.shape[:2]
        small = np.array(Image.fromarray(rgb).resize((WIDTH, HEIGHT)))
        self._ensure_median(small)
        self._buf.append(small)
        if len(self._buf) < self.seq_len or self._median is None:
            return None

        frames = [np.moveaxis(self._median, -1, 0)] if self.bg_mode == "concat" else []
        for f in self._buf:
            frames.append(np.moveaxis(f, -1, 0))
        x = np.concatenate(frames, axis=0)[None] / 255.0
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        pred = self.model(x)[0, -1].cpu().numpy()  # last frame's heatmap, (H,W)
        xy = self._decode(pred)
        return self._smooth(xy)

    def _decode(self, heatmap: np.ndarray) -> Optional[tuple[float, float]]:
        if heatmap.max() <= 0:
            return None
        binmap = (heatmap > self.threshold).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cx, cy = x + w / 2.0, y + h / 2.0
        orig_h, orig_w = self._orig_hw
        return cx * (orig_w / WIDTH), cy * (orig_h / HEIGHT)

    def _smooth(self, xy: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        if xy is None:
            self._misses += 1
            if self._misses > self.seq_len:
                self._kx.reset(); self._ky.reset(); self.trail.clear()
            return None
        self._misses = 0
        sx = self._kx.update(xy[0])
        sy = self._ky.update(xy[1])
        if sx is None or sy is None:
            return None
        self.trail.append((sx, sy))
        if len(self.trail) > self.trail_len:
            self.trail.pop(0)
        return sx, sy
