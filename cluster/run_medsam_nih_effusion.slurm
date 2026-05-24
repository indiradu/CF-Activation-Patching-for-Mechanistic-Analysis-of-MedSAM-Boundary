#!/usr/bin/env bash
#SBATCH -J medsam_nih50
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/medsam_nih50_%j.out
#SBATCH -e logs/medsam_nih50_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MEDSAM_REPO="${MEDSAM_REPO:-$PROJECT_DIR/external/MedSAM}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/checkpoints/medsam_vit_b.pth}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_runtime.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/medsam/nih_effusion_50/original}"
IMAGE_COLUMN="${IMAGE_COLUMN:-image_path}"
PREDICTION_COLUMN="${PREDICTION_COLUMN:-medsam_original_mask_path}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"

nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

PYTHONPATH="$PROJECT_DIR/src:$MEDSAM_REPO" "$PYTHON_BIN" scripts/run_medsam_batch_manifest.py \
  --manifest "$MANIFEST" \
  --medsam-repo "$MEDSAM_REPO" \
  --checkpoint "$CHECKPOINT" \
  --image-column "$IMAGE_COLUMN" \
  --box-column box_xyxy \
  --prediction-column "$PREDICTION_COLUMN" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0
