#!/usr/bin/env bash
# Queue SMALL-variant training for models that benefit from larger capacity.
# Runs AFTER the complete nano chain (queue_all_training.sh + queue_ball_tracknet.sh)
# has finished.  Trains two s-models sequentially:
#   1. player detection (yolo26s) → player_s_best.pt
#   2. body pose        (yolo26s-pose) → bodypose_s_best.pt
#
# Court keypoints already achieves mAP 0.995 with nano — no s-version needed.
# TrackNetV3 ball has no size variant — architecture is fixed.
# Shot classification has small dataset — s-version would not help meaningfully.
#
# Launch manually from the project root:
#   bash scripts/queue_s_training.sh &
#
# The old nano weights in data/models/*_best.pt are NEVER overwritten.
# S-variant outputs have the _s suffix: player_s_best.pt, bodypose_s_best.pt
# ============================================================================
set -uo pipefail
cd /home/tpereira/rep/padel-analytics || { echo "cd failed"; exit 1; }

source .venv/bin/activate

echo "[queue-s] === $(date) waiting for nano chain to finish ==="

# 1. Wait for queue_all_training.sh to exit
while pgrep -f 'queue_all_training.sh' >/dev/null 2>&1; do
  sleep 30
done

# 2. Wait for queue_ball_tracknet.sh to exit
while pgrep -f 'queue_ball_tracknet.sh' >/dev/null 2>&1; do
  sleep 30
done

# 3. Grace period for any straggler YOLO training process
sleep 60
while pgrep -f 'src/train.py' >/dev/null 2>&1; do
  sleep 30
done

echo "[queue-s] nano chain done ($(date)) — starting s-training"

# ── 1. Player detection (small) ──────────────────────────────────────────
echo "[queue-s] === Detection SMALL ($(date)) ==="
: > data/training_detection_s.log
python src/train.py --task detection_combined_s > data/training_detection_s.log 2>&1
echo "[queue-s] Detection SMALL finished ($(date))"
echo "[queue-s] -> data/models/player_s_best.pt (nano model not touched)"

# ── 2. Body pose (small) ────────────────────────────────────────────────
echo "[queue-s] === Body Pose SMALL ($(date)) ==="
: > data/training_bodypose_s.log
python src/train.py --task bodypose_s > data/training_bodypose_s.log 2>&1
echo "[queue-s] Body Pose SMALL finished ($(date))"
echo "[queue-s] -> data/models/bodypose_s_best.pt (nano model not touched)"

echo "[queue-s] ALL S-TRAINING COMPLETE ($(date))"
echo "[queue-s] Models: data/models/player_s_best.pt, bodypose_s_best.pt"
echo "[queue-s] Nano models in data/models/ untouched: player_best court_best bodypose_best shotclass_best ball_best"
