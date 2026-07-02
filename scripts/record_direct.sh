#!/bin/bash
# record_direct.sh - Record video directly to Pi's 2TB SSD via NFS
# Usage: ./record_direct.sh [duration_minutes]
#   duration_minutes: how long to record (default: 60, 0 = infinite)

set -euo pipefail

DURATION=${1:-60}
REC_DIR=/mnt/pi-recordings
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🎾 PadelCV Recording Pipeline${NC}"
echo "Started: $(date)"
echo "Duration: ${DURATION} minutes (${DURATION} = infinite)"
echo "Output: ${REC_DIR}/"

# Check NFS mount
if ! mountpoint -q $REC_DIR; then
    echo -e "${YELLOW}Mounting NFS...${NC}"
    sudo mount $REC_DIR
fi

if ! mountpoint -q $REC_DIR; then
    echo -e "${RED}ERROR: NFS mount at $REC_DIR not available!${NC}"
    echo "Try: sudo mount 192.168.1.112:/mnt/nvme/padelcv/recordings $REC_DIR"
    exit 1
fi

FREE_GB=$(df --output=avail $REC_DIR | tail -1 | awk '{print int($1/1024/1024)}')
echo -e "${GREEN}Pi SSD: ${FREE_GB}GB free${NC}"

# Check camera
if gst-inspect-1.0 nvarguscamerasrc &>/dev/null; then
    CAMERA_SRC="nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,format=I420"
    echo -e "${GREEN}Camera: CSI (nvarguscamerasrc)${NC}"
elif gst-inspect-1.0 v4l2src &>/dev/null && [ -e /dev/video0 ]; then
    CAMERA_SRC="v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080,framerate=30/1"
    echo -e "${YELLOW}Camera: USB (v4l2src /dev/video0)${NC}"
else
    echo -e "${YELLOW}No camera detected. Using test source.${NC}"
    CAMERA_SRC="videotestsrc ! video/x-raw,width=1920,height=1080,framerate=30/1"
fi

# Encode to H.264 (compatible) with segmenting
PIPELINE="$CAMERA_SRC ! queue ! x264enc speed-preset=fast bitrate=8000 !     h264parse ! qtmux !     splitmuxsink location=${REC_DIR}/${TIMESTAMP}_%05d.mp4 max-size-time=60000000000"

echo -e "${GREEN}Starting recording...${NC}"
echo "Pipeline: $PIPELINE"
echo "---"

if [ "$DURATION" -gt 0 ]; then
    timeout $((DURATION * 60)) gst-launch-1.0 -e $PIPELINE
else
    gst-launch-1.0 -e $PIPELINE
fi

echo "---"
echo -e "${GREEN}Recording complete!${NC}"
echo "Files in $REC_DIR:"
ls -lh $REC_DIR/${TIMESTAMP}_* 2>/dev/null
echo ""
echo "Next: Visit https://padel-cv.t-pereira.com to annotate"
echo "      Or https://videos.t-pereira.com to browse"
