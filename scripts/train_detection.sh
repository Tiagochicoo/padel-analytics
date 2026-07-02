#!/usr/bin/env bash
# Train the PLAYER DETECTION model (YOLOv26 detection head).
# Edits: configs/detection.yaml controls base weights + hyperparameters.
# Usage: bash scripts/train_detection.sh [KEY=VALUE ...]
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv if present
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

echo "==> Training player detection (YOLOv26)"
python src/train.py --task detection "$@"

echo "==> Done. Best weights copied to data/models/player_best.pt"
