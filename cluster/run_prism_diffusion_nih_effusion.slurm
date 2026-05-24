#!/usr/bin/env bash
#SBATCH -J prism_nih50
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/cxrsd_nih50_%j.out
#SBATCH -e logs/cxrsd_nih50_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_activation_manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/prism_diffusion/nih_effusion_50/images}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_with_prism_diffusion.csv}"
MODEL_ID="${MODEL_ID:-IrohXu/stable-diffusion-mimic-cxr-v0.1}"
LIMIT="${LIMIT:-10}"
STRENGTH="${STRENGTH:-0.35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
SEED="${SEED:-17}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_MANIFEST")" "$PROJECT_DIR/logs"

nvidia-smi || true

ARGS=(
  scripts/run_prism_diffusion_counterfactuals.py
  --manifest "$MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --output-manifest "$OUTPUT_MANIFEST"
  --model-id "$MODEL_ID"
  --strength "$STRENGTH"
  --guidance-scale "$GUIDANCE_SCALE"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --seed "$SEED"
  --torch-dtype "$TORCH_DTYPE"
  --device cuda
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" "${ARGS[@]}"
