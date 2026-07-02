#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Export trained weights for Jetson Orin Nano (TensorRT).
#
# RUN THIS ON THE JETSON (JetPack + TensorRT) for .engine files.
# On the laptop, build ONNX only (engines are not cross-device portable).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS="${1:-data/models/player_best.pt}"
FORMAT="${2:-engine}"          # 'engine' (on Jetson) or 'onnx' (on laptop)
IMGSZ="${3:-640}"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

echo "==> Exporting ${WEIGHTS} -> ${FORMAT} (imgsz=${IMGSZ})"
python src/export_trt.py \
    --weights "$WEIGHTS" \
    --format "$FORMAT" \
    --imgsz "$IMGSZ" \
    --half \
    --batch 1 \
    --workspace 4

echo "==> Export complete. Deploy the .engine alongside requirements-jetson.txt"
