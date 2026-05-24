#!/usr/bin/env bash
#SBATCH -J actprobe_nih50
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -o logs/actprobe_nih50_%j.out
#SBATCH -e logs/actprobe_nih50_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MEDSAM_REPO="${MEDSAM_REPO:-$PROJECT_DIR/external/MedSAM}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/checkpoints/medsam_vit_b.pth}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_with_counterfactuals.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/activation_probe/nih_effusion_50}"
IMAGE_COLUMN="${IMAGE_COLUMN:-image_path}"
COUNTERFACTUAL_COLUMN="${COUNTERFACTUAL_COLUMN:-counterfactual_path}"
REFERENCE_COLUMN="${REFERENCE_COLUMN:-lung_mask_path}"
LIMIT="${LIMIT:-}"
RUN_PATCHING="${RUN_PATCHING:-1}"
LAYERS="${LAYERS:-block3,block6,block9,neck}"
REGIONS="${REGIONS:-whole,boundary,failure_boundary,missed_boundary,interior,background}"
CONTROLS="${CONTROLS:-matched_cf_to_original}"
NONMATCHED_SHIFT="${NONMATCHED_SHIFT:-1}"
ORIGINAL_PREDICTION_COLUMN="${ORIGINAL_PREDICTION_COLUMN:-medsam_original_mask_path}"
BOUNDARY_WIDTH="${BOUNDARY_WIDTH:-5}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"

nvidia-smi || true

ARGS=(
  scripts/run_medsam_activation_probe.py
  --manifest "$MANIFEST"
  --medsam-repo "$MEDSAM_REPO"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --image-column "$IMAGE_COLUMN"
  --counterfactual-column "$COUNTERFACTUAL_COLUMN"
  --reference-column "$REFERENCE_COLUMN"
  --device cuda:0
  --layers "$LAYERS"
  --regions "$REGIONS"
  --controls "$CONTROLS"
  --nonmatched-shift "$NONMATCHED_SHIFT"
  --original-prediction-column "$ORIGINAL_PREDICTION_COLUMN"
  --boundary-width "$BOUNDARY_WIDTH"
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ "$RUN_PATCHING" == "1" ]]; then
  ARGS+=(--run-patching)
fi

PYTHONPATH="$PROJECT_DIR/src:$MEDSAM_REPO" "$PYTHON_BIN" "${ARGS[@]}"
