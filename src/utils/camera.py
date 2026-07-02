"""
camera.py
=========
Unified video/camera source so inference works identically for files,
webcams and RTSP streams. Centralizing this keeps the inference loop clean
and makes switching to DeepStream later a one-place change.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class FrameSource:
    """Thin wrapper around a cv2.VideoCapture."""
    cap: cv2.VideoCapture
    width: int
    height: int
    fps: float
    is_live: bool
    source: object = None        # original source (int/str) for reconnection

    def read(self):
        """Return (ok, frame)."""
        return self.cap.read()

    def frames(self) -> Iterator:
        """Yield frames until the source ends (live sources never end)."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()


def open_source(source, buffer_size: int = 1) -> FrameSource:
    """
    Open a video file, webcam index (int), or RTSP/HTTP stream.

    Args:
        source: filepath | int camera index | 'rtsp://...' | 'http://...'
        buffer_size: for live sources, 1 = lowest latency (drop stale frames).

    Returns:
        FrameSource ready to iterate.
    """
    is_live = isinstance(source, int) or (isinstance(source, str) and
                                          source.lower().startswith(("rtsp://", "http://", "https://")))

    # Use FFMPEG backend for streams; default for files/webcams.
    cap = cv2.VideoCapture(source)
    if is_live:
        # minimize buffering for real-time responsiveness
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source!r}")

    src = FrameSource(
        cap=cap,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        is_live=is_live,
        source=source,
    )
    print(f"[source] {source} | {src.width}x{src.height} @ {src.fps:.1f} fps "
          f"| live={src.is_live}")
    return src


class ThreadedCapture:
    """Background frame reader for LIVE sources (camera / RTSP).

    Decouples capture from analysis: a daemon thread reads frames continuously
    and keeps only the most recent one, so analysis (which may be slower than
    the camera's 90 fps) never accumulates lag — it always processes the
    freshest frame and stale intermediates are silently dropped. ``latest()``
    blocks until a frame newer than the last consumed one arrives (no busy-spin,
    no reprocessing). Auto-retries on read failure and exposes a ``connected``
    flag for the dashboard.
    """

    def __init__(self, frame_source: FrameSource, reconnect_delay: float = 1.0,
                 reconnect_threshold: int = 10):
        self.src = frame_source
        self.reconnect_delay = reconnect_delay
        self.reconnect_threshold = reconnect_threshold
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest = None                 # (frame, ts)
        self._consumed_ts = 0.0
        self._connected = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)

    @property
    def connected(self) -> bool:
        return self._connected

    def _run(self) -> None:
        consec_fail = 0
        while not self._stop.is_set():
            ok, frame = self.src.read()
            if not ok or frame is None:
                consec_fail += 1
                self._connected = False
                # After N consecutive failures, re-open the device (the cv2
                # socket may be dead and won't self-heal via read() retries).
                if consec_fail >= self.reconnect_threshold:
                    print(f"[camera] {consec_fail} consecutive failures — reconnecting...")
                    try:
                        self.src.release()
                        self.src = open_source(self.src.source)
                        consec_fail = 0
                        print("[camera] reconnected successfully")
                    except Exception as e:
                        print(f"[camera] reconnect failed: {e}")
                if self._stop.wait(self.reconnect_delay):
                    return
                continue
            consec_fail = 0
            self._connected = True
            ts = time.time()
            with self._cond:
                self._latest = (frame, ts)
                self._cond.notify_all()

    def latest(self, block: bool = True, timeout: float = 10.0):
        """Return the most recent frame newer than the last one consumed.

        Blocks up to `timeout` for a fresh frame; returns None on timeout/stop
        (e.g. camera still warming up or reconnecting).
        """
        deadline = time.time() + timeout
        with self._cond:
            while not self._stop.is_set():
                if self._latest is not None and self._latest[1] > self._consumed_ts:
                    frame, ts = self._latest
                    self._consumed_ts = ts
                    return frame
                if not block:
                    return None
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return None
