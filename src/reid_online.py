"""
src/reid_online.py
==================
Online (incremental) Re-ID resolver — stable canonical P1..P4 DURING a match.

The batch resolver (``src/reid.py::run_reid``) needs the whole video, so it
cannot run on a live stream. This resolver instead accumulates BoT-SORT
tracklets frame-by-frame and reuses the batch clustering internals
(``_cluster_tracklets`` + ``_canonicalize``) on the growing prefix during a
warmup phase, then FREEZES at match-start so IDs never swap mid-match.

Lifecycle (driven by ``src/match_state.py``):

    WARMUP  ->  periodically re-cluster (every ``recluster_every`` frames);
                the mapping raw_id -> canonical 1..4 refines as data grows.
                This is the physical pre-match warmup window — low-stakes data.
    LIVE    ->  ``lock_for_match()`` does one final re-cluster, then freezes.
                New raw_ids are assigned to the nearest canonical slot via a
                per-slot position EMA (the same mean-(y,x) key the batch
                canonicalizer uses). No re-clustering -> no mid-match ID swaps.

The expensive appearance extraction (DINOv2/OSNet) only runs during warmup; in
LIVE the per-frame cost is just an EMA update + a nearest-slot lookup, so live
throughput is unaffected.
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

import cv2
import numpy as np

from src.reid import (
    Tracklet, _cluster_tracklets, _canonicalize, build_reid_extractor,
)

_CROP_SIZE = (224, 224)
_POS_EMA_ALPHA = 0.1


class OnlineReIDResolver:
    def __init__(self, device: str = "cpu", k: int = 4, sample_stride: int = 12,
                 max_samples: int = 16, min_tracklet_len: int = 8,
                 recluster_every: int = 600, pos_weight: float = 0.15):
        self.device = device
        self.k = k
        self.sample_stride = sample_stride
        self.max_samples = max_samples
        self.min_tracklet_len = min_tracklet_len
        self.recluster_every = recluster_every
        self.pos_weight = pos_weight

        self._tracklets: dict[int, Tracklet] = {}
        self._sample_count: dict[int, int] = {}
        self._extractor = None
        self.mapping: dict[int, int] = {}          # raw_id -> canonical 1..k
        self._slot_pos: dict[int, np.ndarray] = {} # canonical -> EMA normalised feet
        self.locked = False
        self._frame_idx = 0
        self._last_cluster_at = -10**9

    def update(self, frame: np.ndarray, tracks: list[tuple[int, tuple]],
               frame_idx: int) -> dict[int, int]:
        """Accumulate one frame's (raw_id, box) tracks; return current mapping."""
        self._frame_idx = frame_idx
        if frame is not None and tracks:
            self._accumulate(frame, tracks)
        self._update_slot_pos(tracks, frame)

        if not self.locked:
            due = frame_idx - self._last_cluster_at >= self.recluster_every
            if (not self.mapping or due):
                self._maybe_recluster()
        else:
            for raw_id, _ in tracks:
                if raw_id not in self.mapping:
                    self.mapping[raw_id] = self._nearest_slot(raw_id)
        return self.mapping

    def _accumulate(self, frame: np.ndarray, tracks: list[tuple[int, tuple]]):
        if self.locked:
            return  # no more crops needed once IDs are frozen
        H, W = frame.shape[:2]
        for raw_id, box in tracks:
            t = self._tracklets.get(raw_id)
            if t is None:
                t = Tracklet(track_id=int(raw_id))
                self._tracklets[int(raw_id)] = t
            t.frame_min = min(t.frame_min, self._frame_idx)
            t.frame_max = max(t.frame_max, self._frame_idx)
            sc = self._sample_count.get(raw_id, 0)
            due = (len(t.boxes) == 0
                   or self._frame_idx >= t.boxes[-1][0] + self.sample_stride)
            if sc < self.max_samples and due:
                x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                                  min(W, int(box[2])), min(H, int(box[3])))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    t.crops.append(cv2.resize(crop, _CROP_SIZE))
                    t.boxes.append((self._frame_idx, box))
                    t.feet_pos.append(((box[0] + box[2]) / (2.0 * W), box[3] / H))
                    self._sample_count[raw_id] = sc + 1

    def _maybe_recluster(self) -> None:
        eligible = {tid: t for tid, t in self._tracklets.items()
                    if t.length >= self.min_tracklet_len}
        if len(eligible) < self.k:
            return
        if self._extractor is None:
            self._extractor = build_reid_extractor(device=self.device)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                labels = _cluster_tracklets(eligible, self._extractor,
                                            k=self.k, pos_weight=self.pos_weight)
                raw_to_canon, _ = _canonicalize(labels, eligible)
            self.mapping = {int(r): int(c) for r, c in raw_to_canon.items()}
            self._last_cluster_at = self._frame_idx
        except Exception as e:  # pragma: no cover
            print(f"[reid-online] cluster failed at frame {self._frame_idx}: {e}")

    def _update_slot_pos(self, tracks, frame) -> None:
        if frame is None:
            return
        H, W = frame.shape[:2]
        for raw_id, box in tracks:
            slot = self.mapping.get(int(raw_id))
            if slot is None:
                continue
            feet = np.array([(box[0] + box[2]) / (2.0 * W), box[3] / H], dtype=np.float32)
            cur = self._slot_pos.get(slot)
            self._slot_pos[slot] = feet if cur is None else (1 - _POS_EMA_ALPHA) * cur + _POS_EMA_ALPHA * feet

    def _nearest_slot(self, raw_id: int) -> int:
        t = self._tracklets.get(raw_id)
        if t is not None and t.mean_feet is not None and self._slot_pos:
            pos = np.asarray(t.mean_feet, dtype=np.float32)
            return int(min(self._slot_pos, key=lambda s: float(np.linalg.norm(self._slot_pos[s] - pos))))
        used = set(self.mapping.values())
        for s in range(1, self.k + 1):
            if s not in used:
                return s
        return 1

    def lock_for_match(self) -> None:
        """Freeze: one final re-cluster, then no more clustering (LIVE)."""
        self._maybe_recluster()
        self.locked = True
        # If warmup never converged (not enough tracklets for clustering),
        # fall back to position-based slot assignment so IDs don't pile up.
        if not self.mapping:
            self._position_fallback()
        # Free GPU memory: the extractor is no longer needed after locking.
        self._free_extractor()
        # Clear crop images (the memory-expensive part) but keep tracklet metadata.
        for t in self._tracklets.values():
            t.crops.clear()

    def _position_fallback(self) -> None:
        """When clustering never ran (insufficient warmup data), assign slots
        by mean feet position so players at least get distinct P1..P4 labels."""
        eligible = [(tid, t) for tid, t in self._tracklets.items()
                    if t.mean_feet is not None]
        eligible.sort(key=lambda kv: (kv[1].mean_feet[1], kv[1].mean_feet[0]))
        for i, (raw_id, t) in enumerate(eligible[:self.k]):
            slot = i + 1
            self.mapping[raw_id] = slot
            self._slot_pos[slot] = np.asarray(t.mean_feet, dtype=np.float32)
        print(f"[reid-online] position fallback: {len(self.mapping)} slots assigned")

    def _free_extractor(self) -> None:
        """Move the ReID extractor off GPU to free VRAM for the live match."""
        if self._extractor is None:
            return
        try:
            import torch
            model = getattr(self._extractor, "model", self._extractor)
            if hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass
        self._extractor = None
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
        print("[reid-online] extractor freed from GPU after lock")

    @property
    def status(self) -> dict:
        return {
            "locked": self.locked,
            "tracklets": len(self._tracklets),
            "mapped": len(self.mapping),
            "slots": sorted(self._slot_pos.keys()),
        }


__all__ = ["OnlineReIDResolver"]
