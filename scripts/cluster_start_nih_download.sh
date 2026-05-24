#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/miccai_workshop}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
LOG_FILE="$LOG_DIR/nih_chestxray14_download.log"
PID_FILE="$LOG_DIR/nih_chestxray14_download.pid"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Download already running with PID $old_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

nohup "$PROJECT_DIR/scripts/cluster_download_nih_chestxray14.sh" > "$LOG_FILE" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "Started NIH ChestX-ray14 download with PID $pid"
echo "Log: $LOG_FILE"

