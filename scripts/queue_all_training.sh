#!/usr/bin/env bash
# Unified training queue: court pose -> bodypose -> shotclass
#
# Runs after detection training finishes.  Handles the remaining Ultralytics
# training chain so the GPU is never idle.
#
# NOTE: BALL detection is intentionally NOT trained here. It uses TrackNetV3
# (CenterNet), which cannot use src/train.py. TrackNet ball training is handled
# separately by scripts/queue_ball_tracknet.sh (Phase 1c) after shotclass.
#
# Match strings are INSIDE this file so pgrep never self-matches.
set -uo pipefail
cd /home/tpereira/rep/padel-analytics

source .venv/bin/activate

run_task() {
  local task="$1" logfile="$2" label="$3"
  echo "[queue] === Starting $label ($(date)) ==="
  : > "$logfile"
  python src/train.py --task "$task" > "$logfile" 2>&1
  echo "[queue] === $label finished ($(date)) ==="
}

# ── 1. Wait for detection training to finish ──────────────────────────────
echo "[queue] waiting for detection training (detection_combined)..."
while pgrep -f 'task detection_combined' >/dev/null 2>&1; do
  sleep 30
done
echo "[queue] detection done ($(date))"

# ── 2. Grace period for court pose to start (queue_court_training.sh) ─────
echo "[queue] waiting 90s for court pose to start..."
sleep 90

# ── 3. Wait for court pose to finish ─────────────────────────────────────
if pgrep -f 'task pose' >/dev/null 2>&1; then
  echo "[queue] court pose running, waiting..."
  while pgrep -f 'task pose' >/dev/null 2>&1; do sleep 30; done
  echo "[queue] court pose done ($(date))"
else
  echo "[queue] court pose not detected, proceeding"
fi

# ── 4. Body pose (17 COCO keypoints, full 100k) ──────────────────────────
run_task bodypose  data/training_bodypose.log  "BODY POSE"

# ── 5. Shot classification (11 shot classes) ─────────────────────────────
run_task shotclass data/training_shotclass.log "SHOT CLASSIFICATION"

# Ball detection (TrackNetV3) is trained separately after shotclass by
# scripts/queue_ball_tracknet.sh — it cannot use src/train.py.

echo "[queue] ALL TRAINING COMPLETE ($(date))"
echo "[queue] Models in data/models/: player_best court_best bodypose_best shotclass_best"
echo "[queue] (ball_best is produced later by the TrackNetV3 queue)"
