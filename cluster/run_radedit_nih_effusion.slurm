#!/usr/bin/env bash
#SBATCH -J radedit_nih
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/radedit_nih_%j.out
#SBATCH -e logs/radedit_nih_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_activation_manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/radedit/nih_effusion_50_pilot10/images}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_with_radedit_pilot10.csv}"
LIMIT="${LIMIT:-1}"
PROMPT="${PROMPT:-No acute cardiopulmonary process}"
WEIGHT="${WEIGHT:-7.5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SKIP_RATIO="${SKIP_RATIO:-0.3}"
SEED="${SEED:-17}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_MANIFEST")" "$PROJECT_DIR/logs"

nvidia-smi || true

ARGS=(
  scripts/run_radedit_counterfactuals.py
  --manifest "$MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --output-manifest "$OUTPUT_MANIFEST"
  --prompt "$PROMPT"
  --weight "$WEIGHT"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --skip-ratio "$SKIP_RATIO"
  --seed "$SEED"
  --torch-dtype "$TORCH_DTYPE"
  --device cuda
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" "${ARGS[@]}"
