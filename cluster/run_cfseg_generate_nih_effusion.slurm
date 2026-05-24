#!/usr/bin/env bash
#SBATCH -J cfseg_nih50
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/cfseg_nih50_%j.out
#SBATCH -e logs/cfseg_nih50_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CFSEG_REPO="${CFSEG_REPO:-$PROJECT_DIR/external/CF-Seg}"
CFSEG_INPUT_ROOT="${CFSEG_INPUT_ROOT:-$PROJECT_DIR/data/runtime/cfseg_nih_effusion_50}"
CFSEG_CHECKPOINT_ROOT="${CFSEG_CHECKPOINT_ROOT:-$PROJECT_DIR/checkpoints/cfseg/causal_gen_checkpoints/s_a_v_p_r}"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_DIR/outputs/counterfactuals/cfseg_raw}"
SAVE_FOLDER="${SAVE_FOLDER:-nih_effusion_50_pe_to_no_finding}"

DSCM_CHECKPOINT="${DSCM_CHECKPOINT:-$CFSEG_CHECKPOINT_ROOT/soft_cf_1_0_mimic_train_100_lr_0_001_wd_0_001_lr_lag_0_001_alpha_0_1_beta_3_pe_only/881_checkpoint.pt}"
PGM_CHECKPOINT="${PGM_CHECKPOINT:-$CFSEG_CHECKPOINT_ROOT/pgm_mimic_train_pe_only_100/checkpoint.pt}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-$CFSEG_CHECKPOINT_ROOT/hvae_mimic_train_100_beta_3_lr_0_001_var_0_01_wd_0_001_pe_only/checkpoint.pt}"

cd "$CFSEG_REPO/causal-gen/src/pgm"
mkdir -p "$SAVE_ROOT" "$PROJECT_DIR/logs"

nvidia-smi || true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

PYTHONPATH="$CFSEG_REPO/causal-gen/src:$CFSEG_REPO/causal-gen/src/pgm" "$PYTHON_BIN" save_cf.py \
  --dataset mimic_pe \
  --seed 42 \
  --deterministic \
  --load_path "$DSCM_CHECKPOINT" \
  --save_path "$SAVE_ROOT" \
  --save_folder "$SAVE_FOLDER" \
  --inputpath "$CFSEG_INPUT_ROOT/CheXMask_files_preprocessed" \
  --labelpath "$CFSEG_INPUT_ROOT/CheXMask_segmentation_preprocessed" \
  --csvpath "$CFSEG_INPUT_ROOT/mimic_csv" \
  --train_csv train_pe_only.csv \
  --valid_csv valid_pe_only.csv \
  --test_csv test_pe_only.csv \
  --save_set test \
  --rca_threshold 0.0 \
  --cf_particles 1 \
  --interven_variables pe_finding \
  --pgm_predictor_path "$PGM_CHECKPOINT" \
  --vae_path "$VAE_CHECKPOINT"
