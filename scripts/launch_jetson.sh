#!/usr/bin/env bash
# ============================================================================
# launch_jetson.sh — one-command live deployment on Jetson Orin Nano
# ============================================================================
# Usage:
#   ./scripts/launch_jetson.sh           # foreground (Ctrl-C to stop)
#   ./scripts/launch_jetson.sh --daemon  # background (logs to logs/server.log)
#
# Env vars (override in .env or export before running):
#   CAMERA_INDEX=0        USB camera /dev/videoN index
#   RTSP_URL=             RTSP stream URL (overrides CAMERA_INDEX)
#   VIDEO_SOURCE=         File path (overrides both)
#   INFERENCE_DEVICE=0    GPU (default) or cpu
#   CLEAN_HUD=1           Hide burned-in HUD (premium UI overlay)
#   PORT=8000             Web server port
#   BRAND_NAME=           Rebrandable app name
#   BRAND_TAGLINE=        Subtitle
#   BRAND_ACCENT=         Hex color (default #22d3ee cyan)
#   BRAND_ACCENT2=        Gradient end hex color (default #6366f1 indigo)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env if present (does not override already-exported vars)
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Defaults (only if not already set)
: "${CAMERA_INDEX:=0}"
: "${INFERENCE_DEVICE:=0}"
: "${CLEAN_HUD:=1}"
: "${PORT:=8000}"

export CAMERA_INDEX INFERENCE_DEVICE CLEAN_HUD PORT

mkdir -p logs

# GPU / temp check (Jetson-only; harmless elsewhere)
if command -v tegrastats &>/dev/null; then
    echo "[jetson] GPU temp: $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{printf "%.0f°C\n", $1/1000}')"
fi

# TensorRT engine auto-detection
ENGINES=$(find data/models -name "*.engine" 2>/dev/null | head -5)
if [[ -n "$ENGINES" ]]; then
    echo "[jetson] TensorRT engines detected:"
    echo "$ENGINES" | sed 's/^/  /'
else
    echo "[jetson] No .engine files found — running on .pt (slower)."
    echo "         Run scripts/export_jetson.sh on this device to build FP16 engines."
fi

echo "[jetson] starting server on 0.0.0.0:$PORT"
echo "[jetson] camera=CAMERA_INDEX=$CAMERA_INDEX  device=$INFERENCE_DEVICE"
echo "[jetson] live view:  http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):$PORT/"
echo ""

if [[ "${1:-}" == "--daemon" ]]; then
    nohup python -m src.server.app > logs/server.log 2>&1 &
    SERVER_PID=$!
    echo "[jetson] server started (PID $SERVER_PID), logs → logs/server.log"
    echo "[jetson] stop with: kill $SERVER_PID"
    echo $SERVER_PID > logs/server.pid
else
    exec python -m src.server.app
fi
