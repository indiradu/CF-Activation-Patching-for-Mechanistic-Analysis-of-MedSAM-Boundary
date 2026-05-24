#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/miccai_workshop}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER/miccai_workshop}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
LOG_FILE="$LOG_DIR/chexmask_download.log"
PID_FILE="$LOG_DIR/chexmask_download.pid"

mkdir -p "$LOG_DIR" "$SCRATCH_DIR/chexmask"

if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "CheXMask download already running with PID $old_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

nohup bash "$PROJECT_DIR/scripts/download_chexmask_subset.sh" "$SCRATCH_DIR/chexmask" \
  > "$LOG_FILE" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "Started CheXMask NIH subset download with PID $pid"
echo "Log: $LOG_FILE"
