"""
src/tracknet/
=============
Thin import adapter over the vendored TrackNetV3 codebase
(`third_party/tracknetv3/`, upstream github.com/qaz812345/TrackNetV3, MIT).

We vendor only the weight-compatibility-critical pieces (the TrackNet/InpaintNet
architecture, the WBCELoss, get_model and the training constants) so that the
pretrained checkpoints load with zero key mismatch, and write our own dataset
converter (`scripts/build_tracknet_dataset.py`), fine-tune loop
(`src/train_tracknet.py`) and inference wrapper (`src/ball_tracker.py`) on top.

Importing the rest of our codebase should go through this module:

    from src.tracknet import TrackNet, get_model, WBCELoss, HEIGHT, WIDTH, SIGMA

See third_party/tracknetv3/LICENSE for upstream attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "tracknetv3"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# Re-export the weight-critical pieces (importing triggers parse/pandas/cv2 — all installed).
from model import TrackNet, InpaintNet  # noqa: E402
from utils.general import get_model, HEIGHT, WIDTH, SIGMA  # noqa: E402
from utils.metric import WBCELoss  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = PROJECT_ROOT / "data" / "models" / "tracknet_pretrained" / "ckpts"


def pretrained_tracknet_path() -> Path:
    """Path to the vendored pretrained TrackNet weights (for fine-tuning)."""
    return PRETRAINED_DIR / "TrackNet_best.pt"


def pretrained_inpaintnet_path() -> Path:
    """Path to the vendored pretrained InpaintNet weights (rectification stage)."""
    return PRETRAINED_DIR / "InpaintNet_best.pt"


__all__ = [
    "TrackNet", "InpaintNet", "get_model", "WBCELoss",
    "HEIGHT", "WIDTH", "SIGMA",
    "pretrained_tracknet_path", "pretrained_inpaintnet_path",
]
