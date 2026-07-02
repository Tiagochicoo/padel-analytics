#!/usr/bin/env bash
# Queue BODY POSE training to run after:
#   1. Frame extraction finishes
#   2. Detection training (task detection_combined) finishes
#   3. Court pose training (task pose) finishes
#
# The detection→court-pose handoff has a brief gap where no training runs,
# so we use a grace period to avoid starting body-pose prematurely.
#
# Prerequisites (run BEFORE this script):
#   python scripts/extract_frames.py             # extract video frames as JPG
#   python scripts/convert_coco_pose_to_yolo.py  # convert COCO -> YOLO-pose
#
# Match strings below are INSIDE this file so pgrep never self-matches.
set -uo pipefail
cd /home/tpereira/rep/padel-analytics

# 1. Wait for frame extraction to finish
echo "[queue-bodypose] waiting for frame extraction to finish..."
while pgrep -f 'extract_frames' >/dev/null 2>&1; do
  sleep 15
done
FRAME_COUNT=$(find data/datasets/padeltracker100/frames -name "*.jpg" 2>/dev/null | wc -l)
echo "[queue-bodypose] frames extracted: $FRAME_COUNT"

# 2. Wait for detection training to finish
echo "[queue-bodypose] waiting for detection training (detection_combined)..."
while pgrep -f 'task detection_combined' >/dev/null 2>&1; do
  sleep 30
done
echo "[queue-bodypose] detection training done."

# 3. Grace period for court pose to spin up (queue_court_training.sh launches it)
echo "[queue-bodypose] waiting 90s for court pose to start..."
sleep 90

# 4. Wait for court pose to finish (skip if it never started)
if pgrep -f 'task pose' >/dev/null 2>&1; then
  echo "[queue-bodypose] court pose running, waiting for it to finish..."
  while pgrep -f 'task pose' >/dev/null 2>&1; do
    sleep 30
  done
  echo "[queue-bodypose] court pose done."
else
  echo "[queue-bodypose] court pose not running, proceeding."
fi

# 5. Verify we have enough frames
if [ "$FRAME_COUNT" -lt 50000 ]; then
  echo "[queue-bodypose] ERROR: only $FRAME_COUNT frames (< 50000). Aborting."
  exit 1
fi

# 6. Start body pose training
echo "[queue-bodypose] starting BODY POSE training"
source .venv/bin/activate
: > data/training_bodypose.log
python src/train.py --task bodypose > data/training_bodypose.log 2>&1
echo "[queue-bodypose] body pose training finished. See data/training_bodypose.log"
