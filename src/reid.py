"""
reid.py
=======
Global offline Re-ID pipeline for assigning stable canonical IDs (1..4) to
players throughout a padel match.

Pipeline:
  1. collect_tracklets: run BoT-SORT tracking on a video, sample appearance
     crops per tracklet.
  2. cluster_tracklets: constrained agglomerative clustering → K=4 with
     temporal cannot-link + optional position prior.
  3. canonicalize: relabel clusters into deterministic order.
  4. run_reid: convenience that writes a mapping JSON + optional annotated video.

Backends (auto-selected):
  - OSNet (torchreid) if installed and weights found.
  - DINOv2 (facebookresearch/dinov2) via torch.hub — default.
  - Torchvision ResNet50 ImageNet — fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# ReID extractors
# ─────────────────────────────────────────────────────────────────────────────


class ReIDExtractor:
    """Base interface: embed a list of BGR crops → (N, D) normalized float32."""

    dim: int

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _preprocess(self, bgr, size=224):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        return t


    @classmethod
    def pil_mode(cls) -> bool:
        return True


class DINOv2Extractor(ReIDExtractor):
    """DINOv2-ViT-B/14 via torch.hub."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", source="github"
            )
        except Exception:
            print("[reid] DINOv2 torch.hub load failed, trying cache...")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", source="local", trust_repo=True
            )
        self.model.eval().to(self.device)

        self.dim = 768
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        batch = []
        for c in crops:
            if c is None or c.size == 0:
                c = np.zeros((224, 224, 3), np.uint8)
            t = self._preprocess(c, 224)
            batch.append(t)
        x = torch.stack(batch).to(self.device)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            feats = self.model(x)  # cls token (B, 768)
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy()


class OSNetExtractor(ReIDExtractor):
    """OSNet (torchreid) — used only if torchreid is installed."""

    def __init__(self, weights: str | Path | None = None, device="cuda"):
        import torchreid
        from torchreid.utils import FeatureExtractor

        self.device = device

        kwargs = dict(
            model_name="osnet_x1_0",
            image_size=(256, 128),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            device=device,
            verbose=False,
        )
        if weights and os.path.isfile(weights):
            kwargs["model_path"] = str(weights)
        else:
            kwargs["pretrained"] = True  # may download
        self.ext = FeatureExtractor(**kwargs)
        self.dim = 512

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        prep = []
        for c in crops:
            if c is None or c.size == 0:
                c = np.zeros((256, 128, 3), np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (128, 256))
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
            t = (t - 0.485) / 0.229 if c.shape[-1] != 4 else t  # simple norm
            prep.append(t)
        # FeatureExtractor accepts a list of numpy arrays
        with torch.no_grad():
            feats = self.ext(prep)
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        return feats.astype(np.float32)


class TorchvisionExtractor(ReIDExtractor):
    """ResNet50 ImageNet features (fallback)."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        from torchvision.models import resnet50, ResNet50_Weights

        w = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=w)
        self.model.fc = torch.nn.Identity()  # remove classification head
        self.model.eval().to(self.device)

        self.dim = 2048
        self.preprocess = w.transforms()

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        batch = []
        for c in crops:
            if c is None or c.size == 0:
                c = np.zeros((224, 224, 3), np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(rgb)
            t = self.preprocess(pil_im)
            batch.append(t)
        x = torch.stack(batch).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy()


def build_reid_extractor(backend="auto", device="auto") -> ReIDExtractor:
    """Select best available ReID backbone."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "osnet":
        try:
            return OSNetExtractor(device=device)
        except Exception as e:
            print(f"[reid] OSNet failed ({e}), falling back...")
            backend = "auto"

    if backend in ("auto", "dinov2"):
        try:
            return DINOv2Extractor(device=device)
        except Exception as e:
            print(f"[reid] DINOv2 failed ({e}), falling back to torchvision...")

    return TorchvisionExtractor(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Tracklet data & collection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Tracklet:
    """A continuous BoT-SORT track — collected appearance samples + metadata."""

    track_id: int
    frame_min: int = float("inf")
    frame_max: int = -1
    boxes: list = field(default_factory=list)
    crops: list = field(default_factory=list)
    feet_pos: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return max(self.frame_max - self.frame_min + 1, 0)

    @property
    def mean_feet(self) -> np.ndarray | None:
        if not self.feet_pos:
            return None
        return np.mean(np.array(self.feet_pos), axis=0)


def collect_tracklets(
    video_path: str | Path,
    player_model,
    *,
    tracker_cfg: str = "configs/botsort_reid.yaml",
    sample_stride: int = 12,
    max_samples_per_tracklet: int = 16,
    min_tracklet_len: int = 8,
    crop_size: tuple[int, int] = (224, 224),
) -> dict[int, Tracklet]:
    """
    Single-pass streaming track + sample crops.

    Returns a dict of track_id -> Tracklet objects (only those with >=
    min_tracklet_len frames).
    """
    tracklets: dict[int, Tracklet] = {}
    frame_idx = 0
    # probe video for total frames
    tmp = cv2.VideoCapture(str(video_path))
    total_frames = int(tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp.release()

    generator = player_model.track(
        str(video_path),
        stream=True,
        persist=True,
        tracker=tracker_cfg,
        conf=0.3,
        iou=0.5,
        verbose=False,
        imgsz=640,
    )

    pbar = tqdm(total=total_frames, desc="[reid] tracking", unit="frames")
    sample_count: dict[int, int] = defaultdict(int)

    for result in generator:
        frame = result.orig_img
        if frame is None:
            frame_idx += 1
            pbar.update(1)
            continue
        H, W = frame.shape[:2]

        if result.boxes is None or result.boxes.id is None:
            frame_idx += 1
            pbar.update(1)
            continue

        for box, tid in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy()):
            tid = int(tid)
            if tid not in tracklets:
                tracklets[tid] = Tracklet(track_id=tid)
            t = tracklets[tid]
            t.frame_min = min(t.frame_min, frame_idx)
            t.frame_max = max(t.frame_max, frame_idx)

            # sample crops at stride
            if sample_count[tid] < max_samples_per_tracklet and (
                len(t.boxes) == 0 or frame_idx >= t.boxes[-1][0] + sample_stride
            ):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(x1, 0); y1 = max(y1, 0)
                x2 = min(x2, W); y2 = min(y2, H)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, crop_size)
                    t.crops.append(crop)
                    t.boxes.append((frame_idx, box))
                    t.feet_pos.append(
                        ((box[0] + box[2]) / (2.0 * W), box[3] / H)
                    )
                    sample_count[tid] += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()

    # filter short tracklets
    filtered = {
        tid: t for tid, t in tracklets.items() if t.length >= min_tracklet_len
    }

    print(
        f"[reid] collected {len(filtered)} tracklets "
        f"(filtered out {len(tracklets) - len(filtered)} short)"
    )

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Constrained agglomerative clustering K=4
# ─────────────────────────────────────────────────────────────────────────────


def _cluster_tracklets(
    tracklets: dict[int, Tracklet],
    extractor: ReIDExtractor,
    k: int = 4,
    pos_weight: float = 0.15,
) -> dict[int, int]:
    """
    Agglomerative clustering constrained by temporal cannot-link.

    cannot-link: any two tracklets that co-occur in time (frame ranges overlap)
    cannot be merged because they are different people visible together.

    pos_weight controls how much court position difference discourages
    merging (0 = ignore position). Uses normalized image-feet position.

    Returns {track_id: cluster_label (0..k-1)}.
    """
    ids = list(tracklets.keys())
    n = len(ids)
    if n == 0:
        return {}
    k = min(k, n)
    if k <= 0:
        k = 1

    # 1. extract tracklet-level embeddings
    print(f"[reid] extracting embeddings for {n} tracklets ...")
    id_to_emb: dict[int, np.ndarray] = {}

    # If extractor runs on GPU and we have few crops total, batch all at once.
    all_crops: list[np.ndarray] = []
    idx_map: list[int] = []  # tracklet index -> crop index range
    for tid in ids:
        t = tracklets[tid]
        if t.crops:
            all_crops.extend(t.crops)
        idx_map.append(tid)
    # We'll keep tracklet embedding = mean of sampled crop embeddings, L2-normed.
    # To get per-tracklet mean, we need to know splits. Simpler: embed per
    # tracklet in a loop or batched with offset tracking.
    # Batched: embed all crops, then split by tracklet offsets.
    offsets = [0]
    for tid in ids:
        offsets.append(offsets[-1] + len(tracklets[tid].crops))
    # If some tracklets have zero crops, mean of zero leads to zero vec.
    # Fallback: for tracklet with crops, compute mean; for empty, skip.
    # (all should have crops due to sampling)

    if all_crops:
        all_embs = extractor(all_crops)  # (T, D)
        for i, tid in enumerate(ids):
            if 0 < offsets[i + 1] <= len(all_embs):
                seg = all_embs[offsets[i] : offsets[i + 1]]
                em = seg.mean(axis=0)
                em = em / (np.linalg.norm(em) + 1e-8)
                id_to_emb[tid] = em.astype(np.float32)
            else:
                id_to_emb[tid] = np.zeros(extractor.dim, np.float32)
                id_to_emb[tid][0] = 1.0  # avoid zero norm
    else:
        for tid in ids:
            id_to_emb[tid] = np.zeros(extractor.dim, np.float32)
            id_to_emb[tid][0] = 1.0

    # 2. build distance matrix (appearance + optional position)
    emb_mtx = np.array([id_to_emb[tid] for tid in ids], dtype=np.float32)
    app_D = 1.0 - emb_mtx @ emb_mtx.T  # (n, n) cosine distance

    # position distance (normalized image coords)
    pos_mtx = np.array(
        [
            tracklets[tid].mean_feet if tracklets[tid].mean_feet is not None
            else np.array([0.5, 0.5])
            for tid in ids
        ],
        dtype=np.float32,
    )
    pos_D = np.sqrt(np.sum((pos_mtx[:, None] - pos_mtx[None, :]) ** 2, axis=2))
    pos_D = np.clip(pos_D, 0.0, 1.0)

    D = app_D + pos_weight * pos_D
    np.fill_diagonal(D, np.inf)

    # 3. cannot-link: overlapping frame ranges
    print("[reid] building temporal cannot-link constraints ...")
    cl_set: list[set[int]] = [set() for _ in range(n)]
    ranges = [(tracklets[tid].frame_min, tracklets[tid].frame_max) for tid in ids]
    for i in range(n):
        for j in range(i + 1, n):
            if ranges[i][0] <= ranges[j][1] and ranges[j][0] <= ranges[i][1]:
                cl_set[i].add(j)
                cl_set[j].add(i)

    # 4. constrained agglomerative
    print(f"[reid] clustering {n} tracklets -> {k} clusters ...")
    members: list[list[int]] = [[i] for i in range(n)]
    centroids = emb_mtx.copy()
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    active = [True] * n
    n_active = n

    iteration = n
    while n_active > k and iteration > 0:
        iteration -= 1
        # find closest pair (a,b) with no cannot-link and b not in cl_set[a]
        best_val = np.inf
        best_a = best_b = -1
        # recompute distance for active clusters
        C = centroids[active]
        active_D = 1.0 - C @ C.T
        # add position cost for active
        if pos_weight > 0:
            active_pos = pos_mtx[active]
            active_posD = np.sqrt(
                np.sum((active_pos[:, None] - active_pos[None, :]) ** 2, axis=2)
            )
            active_posD = np.clip(active_posD, 0.0, 1.0)
            active_D = active_D + pos_weight * active_posD
        np.fill_diagonal(active_D, np.inf)

        active_idx = [i for i in range(n) if active[i]]
        m = len(active_idx)

        for ai in range(m):
            a_orig = active_idx[ai]
            for bj in range(ai + 1, m):
                b_orig = active_idx[bj]
                if b_orig in cl_set[a_orig]:
                    continue
                val = active_D[ai, bj]
                if val < best_val:
                    best_val = val
                    best_a = a_orig
                    best_b = b_orig

        if best_a < 0:
            break  # no more mergeable pairs

        # merge best_b into best_a
        members[best_a].extend(members[best_b])
        # update centroid and position
        idxs = members[best_a]
        new_c = emb_mtx[idxs].mean(axis=0)
        new_c = new_c / (np.linalg.norm(new_c) + 1e-8)
        centroids[best_a] = new_c
        new_pos = pos_mtx[idxs].mean(axis=0)
        pos_mtx[best_a] = new_pos
        # cannot-link union
        cl_set[best_a] |= cl_set[best_b]
        # remove cl self-refs
        if best_a in cl_set[best_a]:
            cl_set[best_a].discard(best_a)
        # update others' cl references
        for other in range(n):
            if other == best_a or other == best_b or not active[other]:
                continue
            if best_b in cl_set[other]:
                cl_set[other].discard(best_b)
                cl_set[other].add(best_a)
        cl_set[best_b].clear()
        centroids[best_b] = 0
        active[best_b] = False
        n_active -= 1

    # assign labels
    label: dict[int, int] = {}
    cluster_id = 0
    for orig_idx in range(n):
        if not active[orig_idx]:
            continue
        for member_idx in members[orig_idx]:
            track_id = ids[member_idx]
            label[track_id] = cluster_id
        cluster_id += 1

    # assign inactive (short tracks that were merged) to their cluster
    for orig_idx in range(n):
        if not active[orig_idx]:
            # find which active cluster they were merged into
            for a_orig in range(n):
                if active[a_orig] and orig_idx in members[a_orig]:
                    label[ids[orig_idx]] = label.get(ids[orig_idx], cluster_id)
                    break

    print(f"[reid] clustering done — {len(set(label.values()))} clusters formed")
    return label


def _canonicalize(
    cluster_labels: dict[int, int],
    tracklets: dict[int, Tracklet],
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Deterministic relabel: 1..K ordered by (mean_y, mean_x) of cluster members.

    Returns (raw_to_canonical, canonical_to_cluster_id).
    """
    cluster_to_members: dict[int, list[int]] = defaultdict(list)
    for tid, cid in cluster_labels.items():
        cluster_to_members[cid].append(tid)

    cluster_positions: dict[int, tuple[float, float]] = {}
    for cid, member_ids in cluster_to_members.items():
        ys, xs = [], []
        for tid in member_ids:
            t = tracklets.get(tid)
            if t is not None and t.mean_feet is not None:
                xs.append(t.mean_feet[0])
                ys.append(t.mean_feet[1])
        if ys:
            cluster_positions[cid] = (np.mean(ys), np.mean(xs))
        else:
            cluster_positions[cid] = (0.5, 0.5)

    # sort by (mean_y, mean_x)
    sorted_clusters = sorted(cluster_positions, key=lambda c: cluster_positions[c])

    raw_to_canonical: dict[int, int] = {}
    canonical_to_cluster: dict[int, int] = {}
    for canon, cid in enumerate(sorted_clusters, 1):
        for raw_id in cluster_to_members[cid]:
            raw_to_canonical[raw_id] = canon
        canonical_to_cluster[canon] = cid

    return raw_to_canonical, canonical_to_cluster


# ─────────────────────────────────────────────────────────────────────────────
# Run (full offline Re-ID)
# ─────────────────────────────────────────────────────────────────────────────


def run_reid(
    video_path: str | Path,
    *,
    player_weights: str | Path = "yolo11n.pt",
    court_weights: str | Path | None = None,
    tracker_cfg: str = "configs/botsort_reid.yaml",
    k: int = 4,
    backend: str = "auto",
    device: str = "auto",
    output_dir: str | Path = "data/reid_out",
    render: bool = False,
    render_output: str | Path | None = None,
) -> dict:
    """
    End-to-end: track -> collect -> embed -> cluster -> canonicalize.

    Returns a dict with:
      - mapping: dict[int, int]  (raw_id -> canonical 1..K)
      - canonical_order: list[list[int]]  (groups of raw_ids per canonical id)
      - config: used parameters
    """
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reid] loading player model from {player_weights}")
    player_model = YOLO(str(player_weights))

    court_model = None
    if court_weights and os.path.isfile(court_weights):
        print(f"[reid] loading court model from {court_weights}")
        court_model = YOLO(str(court_weights))

    # 1. collect tracklets
    tracklets = collect_tracklets(
        video_path,
        player_model,
        tracker_cfg=tracker_cfg,
    )

    if not tracklets:
        print("[reid] no valid tracklets found")
        return {"mapping": {}}

    # 2. build extractor
    extractor = build_reid_extractor(backend=backend, device=device)
    print(f"[reid] using {type(extractor).__name__} (dim={extractor.dim})")

    # 3. cluster
    raw_labels = _cluster_tracklets(tracklets, extractor, k=k)

    # 4. canonicalize
    raw_to_canonical, _ = _canonicalize(raw_labels, tracklets)

    # 5. build output mapping
    mapping = dict(sorted(raw_to_canonical.items()))
    canonical_groups: dict[int, list[int]] = defaultdict(list)
    for raw, cn in raw_to_canonical.items():
        canonical_groups[cn].append(raw)

    result = {
        "mapping": {str(k): v for k, v in mapping.items()},
        "canonical_order": [
            {"canonical_id": cn, "raw_ids": raw_ids}
            for cn, raw_ids in sorted(canonical_groups.items())
        ],
        "config": {
            "video": str(video_path),
            "player_weights": str(player_weights),
            "court_weights": str(court_weights) if court_weights else None,
            "backend": type(extractor).__name__,
            "dim": extractor.dim,
            "k": k,
            "tracklets_found": len(tracklets),
        },
    }

    # 6. write JSON
    json_path = output_dir / "reid_mapping.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[reid] mapping written to {json_path}")

    # 7. optionally render annotated video
    if render:
        render_path = Path(render_output) if render_output else output_dir / "reid_annotated.mp4"
        _render_annotations(video_path, mapping, player_model, tracker_cfg, render_path)

    return result


def _render_annotations(
    video_path: str | Path,
    mapping: dict[int, int],
    player_model,
    tracker_cfg: str,
    output_path: str | Path,
):
    """Write annotated video with canonical ids displayed."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[reid] cannot open {video_path} for rendering")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    for codec_hint in ("mp4v", "avc1", "XVID"):
        fcc = cv2.VideoWriter_fourcc(*codec_hint)
        w = cv2.VideoWriter(str(output_path), fcc, fps, (W, H))
        if w.isOpened():
            writer = w
            break
        w.release()

    if writer is None:
        print(
            "[reid] WARNING: no video encoder available — install ffmpeg or use "
            "openCV built with encoding support. Skipping render."
        )
        return

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
    ]

    generator = player_model.track(
        str(video_path),
        stream=True,
        persist=True,
        tracker=tracker_cfg,
        conf=0.3,
        iou=0.5,
        verbose=False,
        imgsz=640,
    )

    pbar = tqdm(desc="[reid] rendering", unit="frames")
    for result in generator:
        frame = result.orig_img
        if frame is None:
            continue
        if result.boxes and result.boxes.id is not None:
            for box, tid in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy()
            ):
                tid_int = int(tid)
                canon = mapping.get(tid_int, 0)
                if canon:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = colors[(canon - 1) % len(colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"P{canon}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
        writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
        print(f"[reid] annotated video -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline Re-ID: track->embed->cluster->canonicalize"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--player",
        default="data/models/player_best.pt",
        help="Player detection weights (default: yolo11n fallback)",
    )
    parser.add_argument(
        "--court", default=None, help="Optional court keypoint weights"
    )
    parser.add_argument(
        "--tracker-cfg", default="configs/botsort_reid.yaml", help="BoT-SORT config"
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Number of players (default: 4)"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "dinov2", "osnet", "torchvision"],
        help="ReID backbone",
    )
    parser.add_argument("--device", default="auto", help="torch device")
    parser.add_argument("--out", default="data/reid_out", help="Output directory")
    parser.add_argument("--render", action="store_true", help="Write annotated video")
    parser.add_argument(
        "--render-output", default=None, help="Annotated video path"
    )

    args = parser.parse_args()

    # resolve player weights fallback
    player_w = args.player
    if not os.path.isfile(player_w):
        candidate = "yolo11n.pt"
        if os.path.isfile(candidate):
            player_w = candidate
        else:
            print(f"[reid] {player_w} not found, using yolo11n.pt via Ultralytics")
            player_w = "yolo11n.pt"

    run_reid(
        video_path=args.video,
        player_weights=player_w,
        court_weights=args.court,
        tracker_cfg=args.tracker_cfg,
        k=args.k,
        backend=args.backend,
        device=args.device,
        output_dir=args.out,
        render=args.render,
        render_output=args.render_output,
    )
