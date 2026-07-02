"""
src/train_tracknet.py
=====================
Fine-tune the pretrained TrackNetV3 ball model (`TrackNet`) on our padel ball
trajectories produced by ``scripts/build_tracknet_dataset.py``.

This is a standalone training loop (it does NOT use Ultralytics / src/train.py,
because TrackNet is a CenterNet-style heatmap regressor, not a YOLO detector).
It builds on the weight-critical pieces vendored under ``third_party/tracknetv3``
(re-exported via ``src.tracknet``): the ``TrackNet`` 2D-UNet, the ``WBCELoss``
and the ``HEIGHT/WIDTH/SIGMA`` conventions. The dataset class mirrors the
upstream ``Shuttlecock_Trajectory_Dataset.__getitem__`` semantics (seq_len
stacked RGB frames + a median background channel, normalised /255, and the
binary-DISK heat-map target of radius SIGMA) so the pretrained checkpoint
transfers cleanly.

Layout consumed (see build_tracknet_dataset.py):
    data/datasets/ball_tracknet/<split>/<rally_id>/frames/{0..N-1}.png
                                                   /label.csv   (Frame,Visibility,X,Y)
                                                   /median.npz  (key 'median', uint8 RGB 288x512)
    data/datasets/ball_tracknet/manifest_<split>.txt

Usage:
    python src/train_tracknet.py --epochs 30 --batch_size 8 --seq_len 8 --bg_mode concat
    # resume: add --resume runs/tracknet/TrackNet_cur.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.tracknet import (
    HEIGHT, WIDTH, SIGMA, get_model, WBCELoss, pretrained_tracknet_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "datasets" / "ball_tracknet"
SAVE_DIR = PROJECT_ROOT / "runs" / "tracknet"


def disk_heatmap(cx: int, cy: int) -> np.ndarray:
    """Replicate the upstream binary-DISK heat-map target (radius SIGMA, value 1)."""
    if cx == 0 and cy == 0:
        return np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    x, y = np.meshgrid(np.linspace(1, WIDTH, WIDTH), np.linspace(1, HEIGHT, HEIGHT))
    hm = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    hm = (hm <= SIGMA ** 2).astype(np.float32)
    return hm.reshape(1, HEIGHT, WIDTH)


class TrackNetBallDataset(Dataset):
    """seq_len consecutive-frame windows over padel ball rallies."""

    def __init__(self, split: str, seq_len: int, bg_mode: str, sliding_step: int,
                 orig_w: int = 1920, orig_h: int = 1080):
        self.split = split
        self.seq_len = seq_len
        self.bg_mode = bg_mode
        self.sliding_step = sliding_step
        self.w_scaler = orig_w / WIDTH
        self.h_scaler = orig_h / HEIGHT
        self._median_cache: dict[str, np.ndarray] = {}

        manifest = DATA_ROOT / f"manifest_{split}.txt"
        rally_ids = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
        self._rallies: dict[int, dict] = {}
        self._windows: list[tuple[int, int]] = []
        for rid, rally_id in enumerate(rally_ids):
            label = pd.read_csv(DATA_ROOT / split / rally_id / "label.csv")
            n = len(label)
            if n < seq_len:
                continue
            self._rallies[rid] = {"rally_id": rally_id, "label": label}
            start = 0
            while start + seq_len <= n:
                self._windows.append((rid, start))
                start += sliding_step

    def __len__(self) -> int:
        return len(self._windows)

    def _median(self, rally_id: str) -> np.ndarray:
        m = self._median_cache.get(rally_id)
        if m is None:
            m = np.load(DATA_ROOT / self.split / rally_id / "median.npz")["median"]
            self._median_cache[rally_id] = m
        return m

    def __getitem__(self, idx: int):
        rid, start = self._windows[idx]
        info = self._rallies[rid]
        rally_id, label = info["rally_id"], info["label"]
        frames_dir = DATA_ROOT / self.split / rally_id / "frames"

        frames, heatmaps = [], []
        for i in range(self.seq_len):
            row = label.iloc[start + i]
            img = np.array(Image.open(frames_dir / f"{start + i}.png").resize((WIDTH, HEIGHT)))
            frames.append(np.moveaxis(img, -1, 0))
            cx = int(row["X"] / self.w_scaler) if row["Visibility"] else 0
            cy = int(row["Y"] / self.h_scaler) if row["Visibility"] else 0
            heatmaps.append(disk_heatmap(cx, cy))

        frames = np.concatenate(frames, axis=0)  # (seq_len*3, H, W)
        if self.bg_mode == "concat":
            med = np.moveaxis(self._median(rally_id), -1, 0)  # (3,H,W)
            frames = np.concatenate([med, frames], axis=0)    # ((seq_len+1)*3, H, W)
        frames = (frames / 255.0).astype(np.float32)
        heatmaps = np.concatenate(heatmaps, axis=0).astype(np.float32)  # (seq_len, H, W)
        return torch.from_numpy(frames), torch.from_numpy(heatmaps)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    return x * lam + x[perm] * (1 - lam), y * lam + y[perm] * (1 - lam), lam


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Pixel accuracy on visible-ball pixels (upstream-style proxy for val acc)."""
    model.eval()
    n_correct = n_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (model(x) > 0.5).float()
        n_correct += int(((pred > 0.5) == (y > 0.5)).float().mean().item() * x.numel())
        n_total += x.numel()
    return n_correct / max(n_total, 1)


def build_model(seq_len: int, bg_mode: str, pretrained: bool):
    model = get_model("TrackNet", seq_len, bg_mode)
    if pretrained:
        ckpt = torch.load(pretrained_tracknet_path(), map_location="cpu")
        ms, un = model.load_state_dict(ckpt["model"], strict=False)
        assert not ms and not un, f"pretrained weight mismatch: missing={ms} unexpected={un}"
        print(f"[tracknet] loaded pretrained weights (val_acc={ckpt.get('max_val_acc')})")
    return model


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[tracknet] device={device} seq_len={args.seq_len} bg_mode={args.bg_mode}")

    train_ds = TrackNetBallDataset("train", args.seq_len, args.bg_mode, sliding_step=1)
    val_ds = TrackNetBallDataset("val", args.seq_len, args.bg_mode, sliding_step=args.seq_len)
    print(f"[tracknet] windows: train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(args.seq_len, args.bg_mode, pretrained=not args.scratch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    criterion = WBCELoss

    start_epoch, best = 0, 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("max_val_acc", 0.0)
        print(f"[tracknet] resumed from epoch {start_epoch}")

    param_dict = {"model_name": "TrackNet", "seq_len": args.seq_len, "bg_mode": args.bg_mode}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if args.alpha > 0:
                x, y, _ = mixup(x, y, args.alpha)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if args.verbose and bi % 50 == 0:
                print(f"  ep{epoch} b{bi}/{len(train_loader)} loss={loss.item():.4f}")
        if scheduler:
            scheduler.step()

        val_acc = evaluate(model, val_loader, device) if val_loader.batch_size else 0.0
        avg = running / max(len(train_loader), 1)
        improved = val_acc > best
        best = max(best, val_acc)
        print(f"[tracknet] epoch {epoch}: train_loss={avg:.4f} val_acc={val_acc:.4f} best={best:.4f}")

        def _save(name: str):
            torch.save({
                "epoch": epoch, "max_val_acc": best, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "param_dict": param_dict,
            }, SAVE_DIR / name)

        _save("TrackNet_cur.pt")
        if improved:
            _save("TrackNet_best.pt")

    print(f"[tracknet] done. best val_acc={best:.4f} -> {SAVE_DIR}/TrackNet_best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--bg_mode", default="concat", choices=["", "subtract", "subtract_concat", "concat"])
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--lr_scheduler", default="", choices=["", "StepLR"])
    ap.add_argument("--alpha", type=float, default=0.5, help="batch mixup (Beta); -1=off")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="0", help="'0' for cuda, 'cpu' to force CPU")
    ap.add_argument("--scratch", action="store_true", help="train from scratch (skip pretrained)")
    ap.add_argument("--resume", default="", help="checkpoint path to resume from")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.alpha < 0:
        args.alpha = 0
    train(args)


if __name__ == "__main__":
    main()
