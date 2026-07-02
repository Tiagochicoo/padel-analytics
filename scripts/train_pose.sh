#!/usr/bin/env bash
# Train the COURT KEYPOINTS model (YOLOv26 pose head).
# Edits: configs/pose.yaml controls base weights + kpt_shape + hyperparameters.
# Usage: bash scripts/train_pose.sh [KEY=VALUE ...]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

echo "==> Training court keypoints (YOLOv26-pose)"
python src/train.py --task pose "$@"

echo "==> Done. Best weights copied to data/models/court_best.pt"
