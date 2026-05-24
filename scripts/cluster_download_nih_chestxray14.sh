#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/miccai_workshop}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER/miccai_workshop}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

mkdir -p "$PROJECT_DIR/scripts" "$SCRATCH_DIR/kagglehub" "$LOG_DIR"

PIP_FLAGS=()
if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  PY="$VENV_DIR/bin/python"
elif python3 -m venv "$VENV_DIR"; then
  PY="$VENV_DIR/bin/python"
else
  echo "python3 -m venv is unavailable; falling back to user-site pip install."
  PY="python3"
  PIP_FLAGS=(--user)
fi

"$PY" -m pip install "${PIP_FLAGS[@]}" kagglehub

export KAGGLEHUB_CACHE="$SCRATCH_DIR/kagglehub"
"$PY" "$PROJECT_DIR/scripts/download_nih_chestxray14.py" \
  --cache-dir "$SCRATCH_DIR/kagglehub"
