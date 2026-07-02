"""
src/trt_infer.py
================
Lightweight TensorRT inference for YOLO engines on Jetson — runs GPU inference
WITHOUT PyTorch CUDA. Uses ctypes for GPU memory management and the TensorRT
Python API for engine execution.

Drop-in replacement for Ultralytics YOLO() when a .engine file is available.
Falls back automatically: if _try_load() finds a .engine, it wraps it in a
TRTDetector; otherwise it uses the Ultralytics .pt path as before.

Supported models:
    - Detection (player): output shape (1, 300, 6) — [x1,y1,x2,y2,conf,cls]
    - Pose (court):       output shape (1, 300, 4+1+1+num_kpts*3)

Usage:
    detector = TRTDetector("data/models/player_best.engine")
    boxes = detector.detect(frame)   # → [(x1,y1,x2,y2,conf,cls), ...]
"""
from __future__ import annotations

import ctypes
from typing import Optional

import cv2
import numpy as np

_libcudart = ctypes.CDLL("libcudart.so")
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_libcudart.cudaMalloc.restype = ctypes.c_int
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libcudart.cudaMemcpy.restype = ctypes.c_int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
_libcudart.cudaFree.restype = ctypes.c_int
_libcudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcudart.cudaStreamCreate.restype = ctypes.c_int
_libcudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_libcudart.cudaStreamSynchronize.restype = ctypes.c_int
_H2D, _D2H = 1, 2


def _cuda_malloc(size: int) -> int:
    ptr = ctypes.c_void_p()
    _libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    return ptr.value


def _cuda_free(ptr: int) -> None:
    _libcudart.cudaFree(ctypes.c_void_p(ptr))


def _cuda_memcpy_h2d(dst_gpu: int, src_np: np.ndarray) -> None:
    _libcudart.cudaMemcpy(
        ctypes.c_void_p(dst_gpu), src_np.ctypes.data,
        ctypes.c_size_t(src_np.nbytes), ctypes.c_int(_H2D),
    )


def _cuda_memcpy_d2h(dst_np: np.ndarray, src_gpu: int) -> None:
    _libcudart.cudaMemcpy(
        dst_np.ctypes.data, ctypes.c_void_p(src_gpu),
        ctypes.c_size_t(dst_np.nbytes), ctypes.c_int(_D2H),
    )


class TRTDetector:
    """Minimal TensorRT YOLO detector — GPU inference without PyTorch CUDA."""

    def __init__(self, engine_path: str, conf_thresh: float = 0.35,
                 imgsz: int = 640):
        import tensorrt as trt
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = engine.create_execution_context()

        # I/O tensor info
        self.input_name = engine.get_tensor_name(0)
        self.output_name = engine.get_tensor_name(1)
        in_shape = tuple(engine.get_tensor_shape(self.input_name))
        out_shape = tuple(engine.get_tensor_shape(self.output_name))
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Allocate GPU buffers
        in_bytes = int(np.prod(in_shape) * 4)   # float32
        out_bytes = int(np.prod(out_shape) * 4)
        self.in_gpu = _cuda_malloc(in_bytes)
        self.out_gpu = _cuda_malloc(out_bytes)
        self._out_np = np.empty(out_shape, dtype=np.float32)

        # CUDA stream for async execution
        self._stream = ctypes.c_void_p()
        _libcudart.cudaStreamCreate(ctypes.byref(self._stream))

        # Set tensor addresses
        self.ctx.set_tensor_address(self.input_name, self.in_gpu)
        self.ctx.set_tensor_address(self.output_name, self.out_gpu)

        print(f"[trt] loaded {engine_path.split('/')[-1]} "
              f"in={in_shape} out={out_shape}")

    def _letterbox(self, frame: np.ndarray):
        """Resize with letterbox padding to imgsz×imgsz. Returns (blob, ratio, pad)."""
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(h * r), int(w * r)
        resized = cv2.resize(frame, (nw, nh))
        pad_h = self.imgsz - nh
        pad_w = self.imgsz - nw
        top, left = pad_h // 2, pad_w // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, (left, top)

    def detect(self, frame: np.ndarray):
        """Run detection on a BGR frame. Returns numpy array (N, 6) of
        [x1, y1, x2, y2, conf, cls] in original image coordinates."""
        # Pre-process: letterbox → normalize → CHW → contiguous
        canvas, ratio, (pad_x, pad_y) = self._letterbox(frame)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis])

        # Copy input to GPU
        _cuda_memcpy_h2d(self.in_gpu, blob)

        # Execute
        self.ctx.execute_async_v3(self._stream.value)
        _libcudart.cudaStreamSynchronize(self._stream)

        # Copy output to CPU
        _cuda_memcpy_d2h(self._out_np, self.out_gpu)

        # Post-process
        raw = self._out_np[0]  # (300, 6)
        mask = raw[:, 4] >= self.conf_thresh
        dets = raw[mask]
        if len(dets) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Scale from letterbox back to original image
        dets[:, 0] = (dets[:, 0] - pad_x) / ratio
        dets[:, 2] = (dets[:, 2] - pad_x) / ratio
        dets[:, 1] = (dets[:, 1] - pad_y) / ratio
        dets[:, 3] = (dets[:, 3] - pad_y) / ratio
        return dets

    def detect_pose(self, frame: np.ndarray):
        """Run pose detection. Returns (N, cols) where cols = 4+1+1+num_kpts*3.
        Box values are [x1,y1,x2,y2,conf,cls] followed by [kx,ky,kvis]*num_kpts."""
        canvas, ratio, (pad_x, pad_y) = self._letterbox(frame)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis])

        _cuda_memcpy_h2d(self.in_gpu, blob)
        self.ctx.execute_async_v3(self._stream.value)
        _libcudart.cudaStreamSynchronize(self._stream)
        _cuda_memcpy_d2h(self._out_np, self.out_gpu)

        raw = self._out_np[0]
        mask = raw[:, 4] >= self.conf_thresh
        dets = raw[mask]
        if len(dets) == 0:
            return np.empty((0, self.out_shape[-1]), dtype=np.float32)

        dets[:, 0] = (dets[:, 0] - pad_x) / ratio
        dets[:, 2] = (dets[:, 2] - pad_x) / ratio
        dets[:, 1] = (dets[:, 1] - pad_y) / ratio
        dets[:, 3] = (dets[:, 3] - pad_y) / ratio
        # Scale keypoint x,y (every 3rd starting from index 6)
        for i in range(6, dets.shape[1], 3):
            dets[:, i] = (dets[:, i] - pad_x) / ratio
            dets[:, i + 1] = (dets[:, i + 1] - pad_y) / ratio
        return dets


__all__ = ["TRTDetector", "SimpleTracker"]


class SimpleTracker:
    """Centroid-distance tracker for TRTDetector output.
    Assigns stable IDs by matching detection centroids frame-to-frame."""

    def __init__(self, max_dist: float = 80, max_miss: int = 10):
        self.max_dist = max_dist
        self.max_miss = max_miss
        self._next_id = 1
        self._tracks: dict[int, dict] = {}  # id → {box, miss, last_seen}

    def update(self, dets: np.ndarray):
        """Takes (N,6) detections. Returns list of (track_id, box)."""
        results = []
        used = set()
        # Match existing tracks to new detections
        for tid, trk in list(self._tracks.items()):
            cx_t = (trk["box"][0] + trk["box"][2]) / 2
            cy_t = (trk["box"][1] + trk["box"][3]) / 2
            best_i, best_d = -1, self.max_dist
            for i, d in enumerate(dets):
                if i in used:
                    continue
                cx_d = (d[0] + d[2]) / 2
                cy_d = (d[1] + d[3]) / 2
                dist = ((cx_t - cx_d) ** 2 + (cy_t - cy_d) ** 2) ** 0.5
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i >= 0:
                used.add(best_i)
                trk["box"] = tuple(dets[best_i][:4])
                trk["miss"] = 0
                results.append((tid, trk["box"]))
            else:
                trk["miss"] += 1
                if trk["miss"] > self.max_miss:
                    del self._tracks[tid]
                else:
                    results.append((tid, trk["box"]))

        # New detections → new IDs
        for i, d in enumerate(dets):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"box": tuple(d[:4]), "miss": 0}
            results.append((tid, tuple(d[:4])))
        return results
