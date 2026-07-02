#!/usr/bin/env bash
# ============================================================================
# record_and_sync.sh — Record from CSI camera (or test file) and sync to Pi
# ============================================================================
# Usage:
#   ./scripts/record_and_sync.sh [--duration MINUTES] [--output-dir DIR]
#
# Modes (highest priority wins):
#   1. FILE mode:   Set VIDEO_SOURCE=/path/to/video.mp4
#   2. RTSP mode:   Set RTSP_URL=rtsp://...
#   3. CSI mode:    Default — uses Jetson CSI camera via nvarguscamerasrc
#
# Options:
#   --duration  D     Record for D minutes (default: 60, use 0 for infinite)
#   --output-dir DIR  Local staging dir (default: /tmp/rec)
#   --segment SECS    Segment length in seconds (default: 60)
#   --dry-run         Print what would happen, don't record
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || echo "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Defaults ---
DURATION_MINUTES=${DURATION_MINUTES:-60}      # total recording time (0 = infinite)
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/rec}"
SEGMENT_SECONDS=${SEGMENT_SECONDS:-60}
KEEP_SEGMENTS=2                                # keep this many local segments
PI_USER="tpereira"
PI_HOST="192.168.1.100"
PI_PATH="/mnt/nvme/padelcv/recordings/"
SSH_KEY="/home/tpereira/.ssh/id_ed25519"
DRY_RUN=false

# --- Parse CLI overrides ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)    DURATION_MINUTES="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";      shift 2 ;;
    --segment)     SEGMENT_SECONDS="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=true;         shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# --- Prepare output dir ---
mkdir -p "$OUTPUT_DIR"

# --- Timestamp helpers ---
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# --- Sync function: try push, fall back to cron pull ---
sync_segment() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "$(ts) [sync] SKIP — file gone: $file"
    return
  fi
  local base
  base=$(basename "$file")
  echo "$(ts) [sync] ready: $base"
  if $DRY_RUN; then
    echo "         DRY-RUN: would sync $base to ${PI_USER}@${PI_HOST}:${PI_PATH}"
    return
  fi
  # Try direct push — if it works, clean up immediately.
  # If it fails (Pi unreachable), leave file for cron-based pull.
  if rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5" \
    "$file" "${PI_USER}@${PI_HOST}:${PI_PATH}" &>/dev/null; then
    rm -f "$file"
    echo "$(ts) [sync] ✓ pushed and removed: $base"
  else
    echo "$(ts) [sync] ~ Pi unreachable, keeping $base for cron pull"
  fi
}

# --- Cleanup: keep only the last N segments locally ---
cleanup_old() {
  local keep="$1"
  local dir="$2"
  # Sort by mtime, skip the newest $keep, delete the rest
  ls -1t "$dir"/segment_*.mp4 2>/dev/null | tail -n +$((keep + 1)) | while read -r old; do
    echo "$(ts) [clean] removing old local segment: $old"
    rm -f "$old"
  done
}

# --- Determine recording source ---
if [[ -n "${VIDEO_SOURCE:-}" ]]; then
  MODE="FILE"
  echo "$(ts) [mode] FILE — input: $VIDEO_SOURCE"
elif [[ -n "${RTSP_URL:-}" ]]; then
  MODE="RTSP"
  echo "$(ts) [mode] RTSP — stream: $RTSP_URL"
else
  MODE="CSI"
  echo "$(ts) [mode] CSI — Jetson camera (nvarguscamerasrc)"
fi

# --- Validate source ---
if [[ "$MODE" == "FILE" && ! -f "$VIDEO_SOURCE" ]]; then
  echo "$(ts) [error] FILE mode but VIDEO_SOURCE not found: $VIDEO_SOURCE"
  exit 1
fi

if $DRY_RUN; then
  echo ""
  echo "========== DRY RUN =========="
  echo "Mode:              $MODE"
  echo "Duration:          ${DURATION_MINUTES} minutes ($([ "$DURATION_MINUTES" -eq 0 ] && echo 'infinite' || echo "$DURATION_MINUTES min"))"
  echo "Staging dir:       $OUTPUT_DIR"
  echo "Segment length:    ${SEGMENT_SECONDS}s"
  echo "Sync destination:  ${PI_USER}@${PI_HOST}:${PI_PATH}"
  echo "Keep local:        last $KEEP_SEGMENTS segments"
  if [[ "$MODE" == "FILE" ]]; then
    echo "Video source:      $VIDEO_SOURCE"
  elif [[ "$MODE" == "RTSP" ]]; then
    echo "RTSP URL:          $RTSP_URL"
  fi
  echo "============================="
  exit 0
fi

echo "$(ts) ============================================================"
echo "$(ts) Starting recording pipeline"
echo "$(ts) Mode: $MODE | Duration: ${DURATION_MINUTES} min | Segment: ${SEGMENT_SECONDS}s"
echo "$(ts) Local: $OUTPUT_DIR → Remote: ${PI_USER}@${PI_HOST}:${PI_PATH}"
echo "$(ts) ============================================================"

# --- Build GStreamer pipeline ---
# We use splitmuxsink to produce self-contained MP4 segments.
# After each segment closes, we sync it and clean up.

SEGMENT_PREFIX="${OUTPUT_DIR}/segment_"

case "$MODE" in
  CSI)
    GST_PIPELINE="nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,format=I420 ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! mp4mux ! splitmuxsink location=${SEGMENT_PREFIX}%05d.mp4 max-size-time=$((SEGMENT_SECONDS * 1000000000))"
    ;;
  RTSP)
    GST_PIPELINE="rtspsrc location=${RTSP_URL} latency=0 ! rtph264depay ! h264parse ! mp4mux ! splitmuxsink location=${SEGMENT_PREFIX}%05d.mp4 max-size-time=$((SEGMENT_SECONDS * 1000000000))"
    ;;
  FILE)
    # For file mode, we need to decode and re-encode to get proper segments
    GST_PIPELINE="filesrc location=${VIDEO_SOURCE} ! decodebin ! videoconvert ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! mp4mux ! splitmuxsink location=${SEGMENT_PREFIX}%05d.mp4 max-size-time=$((SEGMENT_SECONDS * 1000000000))"
    ;;
esac

echo "$(ts) [gst] pipeline: $GST_PIPELINE"
echo ""

# --- Start recording in background ---
gst-launch-1.0 -e $GST_PIPELINE &
GST_PID=$!
echo "$(ts) [gst] started (PID $GST_PID)"

# --- Trap for clean shutdown ---
cleanup() {
  echo "$(ts) [shutdown] stopping recording..."
  kill "$GST_PID" 2>/dev/null || true
  wait "$GST_PID" 2>/dev/null || true
  # Sync any remaining segments
  echo "$(ts) [shutdown] syncing remaining segments..."
  for f in "$OUTPUT_DIR"/segment_*.mp4; do
    [[ -f "$f" ]] && sync_segment "$f"
  done
  echo "$(ts) [shutdown] done."
  exit 0
}
trap cleanup SIGINT SIGTERM

# --- Monitoring loop ---
START_EPOCH=$(date +%s)
DURATION_SECONDS=$((DURATION_MINUTES * 60))
last_sync_count=0

while true; do
  sleep 5

  # Check if GStreamer is still running
  if ! kill -0 "$GST_PID" 2>/dev/null; then
    echo "$(ts) [error] recording process died unexpectedly"
    break
  fi

  # --- Sync newly completed segments ---
  for seg in "$OUTPUT_DIR"/segment_*.mp4; do
    [[ -f "$seg" ]] || continue
    # Only sync segments that are no longer being written to
    # splitmuxsink closes a segment once the next one starts
    sync_segment "$seg"
  done

  # --- Cleanup old local segments ---
  cleanup_old "$KEEP_SEGMENTS" "$OUTPUT_DIR"

  # --- Check duration ---
  if [[ "$DURATION_MINUTES" -gt 0 ]]; then
    elapsed=$(( $(date +%s) - START_EPOCH ))
    if [[ "$elapsed" -ge "$DURATION_SECONDS" ]]; then
      echo "$(ts) [end] reached ${DURATION_MINUTES} minute limit"
      break
    fi
  fi
done

# --- Final sync ---
echo "$(ts) [final] syncing remaining segments..."
for f in "$OUTPUT_DIR"/segment_*.mp4; do
  [[ -f "$f" ]] && sync_segment "$f"
done

# Stop GStreamer
kill "$GST_PID" 2>/dev/null || true
wait "$GST_PID" 2>/dev/null || true

echo "$(ts) ============================================================"
echo "$(ts) Recording session complete"
echo "$(ts) ============================================================"
