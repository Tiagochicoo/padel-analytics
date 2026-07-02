#!/usr/bin/env bash
# Queue TrackNetV3 ball fine-tuning to run AFTER the whole YOLO chain
# (queue_all_training.sh: court -> bodypose -> shotclass) has finished.
#
# TrackNetV3 is NOT a YOLO model and cannot use src/train.py; it runs
# src/train_tracknet.py on the dataset built by scripts/build_tracknet_dataset.py.
# Outputs runs/tracknet/TrackNet_best.pt and promotes it to data/models/ball_best.pt
# so PadelAnalyzer auto-loads it.
#
# Launch detached (survives logout). Match strings are INSIDE this file so the
# watcher's own process is never matched by pgrep.
set -uo pipefail
cd /home/tpereira/rep/padel-analytics

echo "[queue-ball] waiting for the YOLO training chain to finish..."
# 1. If the unified YOLO queue is running, wait for it to exit.
while pgrep -f 'queue_all_training.sh' >/dev/null 2>&1; do
  sleep 30
done
# 2. Grace period + guard against a stray src/train.py (manual run / overlap).
sleep 30
while pgrep -f 'src/train.py' >/dev/null 2>&1; do
  sleep 30
done
echo "[queue-ball] YOLO chain done ($(date))"

source .venv/bin/activate

# 3. Ensure the TrackNet dataset exists (CPU, idempotent).
if [ ! -f data/datasets/ball_tracknet/manifest_train.txt ]; then
  echo "[queue-ball] building TrackNet dataset..."
  python scripts/build_tracknet_dataset.py || { echo "[queue-ball] dataset build FAILED"; exit 1; }
fi

# 4. Fine-tune TrackNetV3 from the pretrained shuttlecock weights.
echo "[queue-ball] starting TrackNetV3 ball fine-tuning ($(date))"
: > data/training_ball.log
python src/train_tracknet.py \
    --epochs 30 --batch_size 8 --seq_len 8 --bg_mode concat \
    --alpha 0.5 --workers 4 \
    > data/training_ball.log 2>&1
echo "[queue-ball] training finished ($(date))"

# 5. Promote best weights into the analyzer's expected path.
if [ -f runs/tracknet/TrackNet_best.pt ]; then
  cp runs/tracknet/TrackNet_best.pt data/models/ball_best.pt
  echo "[queue-ball] -> data/models/ball_best.pt"
else
  echo "[queue-ball] WARNING: runs/tracknet/TrackNet_best.pt not produced"
fi
echo "[queue-ball] done ($(date))"
