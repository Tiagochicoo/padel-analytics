#!/usr/bin/env bash
# Train the BODY POSE model (YOLOv26-pose, 17 COCO keypoints).
# Edits: configs/bodypose.yaml controls base weights + kpt_shape + hyperparameters.
# Usage: bash scripts/train_bodypose.sh [KEY=VALUE ...]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

echo "==> Training body pose (YOLOv26-pose, 17 COCO keypoints)"
python src/train.py --task bodypose "$@"
echo "==> Done. Best weights copied to data/models/bodypose_best.pt"
