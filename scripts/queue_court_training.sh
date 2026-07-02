#!/usr/bin/env bash
# Wait for the combined PLAYER training to finish, then train the COURT
# keypoint (pose) model. Launched detached so it survives.
#
# Match string below is INSIDE this file (not in the launcher's argv), so the
# watcher's own process is never matched by pgrep.
set -uo pipefail
cd /home/tpereira/rep/padel-analytics

echo "[queue] waiting for player training (task detection_combined) to finish..."
while pgrep -f 'task detection_combined' >/dev/null 2>&1; do
  sleep 30
done

echo "[queue] player training done -> starting COURT (pose) training"
source .venv/bin/activate
: > data/training_pose.log
python src/train.py --task pose > data/training_pose.log 2>&1
echo "[queue] court training finished. See data/training_pose.log"
