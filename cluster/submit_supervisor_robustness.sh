#!/usr/bin/env bash
# Submit supervisor-requested robustness experiments for the NIH effusion 50 cohort.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MANIFEST="${BASE_MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_activation_manifest.csv}"
PRISM_MANIFEST="${PRISM_MANIFEST:-$PROJECT_DIR/outputs/medsam/nih_effusion_50/prism_diffusion_v2/manifest_with_predictions.csv}"

cd "$PROJECT_DIR"
mkdir -p logs data/manifests

timestamp="$(date +%Y%m%d_%H%M%S)"
job_log="$PROJECT_DIR/logs/supervisor_robustness_jobs_${timestamp}.txt"
touch "$job_log"

submit() {
  local label="$1"
  shift
  local job_id
  job_id="$("$@")"
  printf "%s,%s\n" "$label" "$job_id" | tee -a "$job_log"
}

echo "Writing job ids to $job_log"

for pct_label in 05 10 15; do
  pct="0.${pct_label}"
  jitter_manifest="$PROJECT_DIR/data/manifests/nih_effusion_50_prompt_jitter_${pct_label}.csv"
  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/create_prompt_jitter_manifest.py \
    --manifest "$BASE_MANIFEST" \
    --output "$jitter_manifest" \
    --pct "$pct" \
    --mode expand_translate

  submit "medsam_jitter_${pct_label}_original" \
    sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",PYTHON_BIN="$PYTHON_BIN",MANIFEST="$jitter_manifest",OUTPUT_DIR="$PROJECT_DIR/outputs/medsam/nih_effusion_50/prompt_jitter_${pct_label}/original",IMAGE_COLUMN=image_path,PREDICTION_COLUMN=medsam_original_jitter_mask_path \
      cluster/run_medsam_nih_effusion.slurm

  submit "medsam_jitter_${pct_label}_cfseg" \
    sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",PYTHON_BIN="$PYTHON_BIN",MANIFEST="$jitter_manifest",OUTPUT_DIR="$PROJECT_DIR/outputs/medsam/nih_effusion_50/prompt_jitter_${pct_label}/cfseg",IMAGE_COLUMN=counterfactual_path,PREDICTION_COLUMN=medsam_cfseg_jitter_mask_path \
      cluster/run_medsam_nih_effusion.slurm

  submit "activation_prompt_jitter_${pct_label}_cfseg" \
    sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",PYTHON_BIN="$PYTHON_BIN",MANIFEST="$jitter_manifest",OUTPUT_DIR="$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prompt_jitter_${pct_label}",COUNTERFACTUAL_COLUMN=counterfactual_path,LAYERS=block3,block9,neck,REGIONS=whole,boundary,background,CONTROLS=matched_cf_to_original,reverse_original_to_cf,nonmatched_cf_to_original,BOUNDARY_WIDTH=32 \
      cluster/run_activation_probe_nih_effusion.slurm
done

for width in 16 64; do
  submit "activation_cfseg_bw${width}_matched" \
    sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",PYTHON_BIN="$PYTHON_BIN",MANIFEST="$BASE_MANIFEST",OUTPUT_DIR="$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw${width}",COUNTERFACTUAL_COLUMN=counterfactual_path,LAYERS=block3,block9,neck,REGIONS=whole,boundary,background,random_like_boundary,random_like_background,CONTROLS=matched_cf_to_original,BOUNDARY_WIDTH="$width" \
      cluster/run_activation_probe_nih_effusion.slurm
done

for width in 16 32 64; do
  submit "activation_cxrsd_bw${width}_random_matched" \
    sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",PYTHON_BIN="$PYTHON_BIN",MANIFEST="$PRISM_MANIFEST",OUTPUT_DIR="$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw${width}",COUNTERFACTUAL_COLUMN=prism_counterfactual_path,LAYERS=block3,block9,neck,REGIONS=whole,boundary,background,random_like_boundary,random_like_background,CONTROLS=matched_cf_to_original,BOUNDARY_WIDTH="$width" \
      cluster/run_activation_probe_nih_effusion.slurm
done

echo "Submitted supervisor robustness jobs."
echo "$job_log"
