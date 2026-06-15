#!/usr/bin/env bash
# Rerun supervisor robustness activation jobs with comma-valued env vars safely exported.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MANIFEST="${BASE_MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_activation_manifest.csv}"
PRISM_MANIFEST="${PRISM_MANIFEST:-$PROJECT_DIR/outputs/medsam/nih_effusion_50/prism_diffusion_v2/manifest_with_predictions.csv}"

cd "$PROJECT_DIR"
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
job_log="$PROJECT_DIR/logs/supervisor_robustness_missing_regions_jobs_${timestamp}.txt"
touch "$job_log"

submit_activation() {
  local label="$1"
  local manifest="$2"
  local output_dir="$3"
  local counterfactual_column="$4"
  local layers="$5"
  local regions="$6"
  local controls="$7"
  local boundary_width="$8"
  local job_id

  job_id="$(
    PROJECT_DIR="$PROJECT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    MANIFEST="$manifest" \
    OUTPUT_DIR="$output_dir" \
    COUNTERFACTUAL_COLUMN="$counterfactual_column" \
    LAYERS="$layers" \
    REGIONS="$regions" \
    CONTROLS="$controls" \
    BOUNDARY_WIDTH="$boundary_width" \
    sbatch --parsable --export=ALL cluster/run_activation_probe_nih_effusion.slurm
  )"
  printf "%s,%s\n" "$label" "$job_id" | tee -a "$job_log"
}

echo "Writing job ids to $job_log"

for pct_label in 05 10 15; do
  jitter_manifest="$PROJECT_DIR/data/manifests/nih_effusion_50_prompt_jitter_${pct_label}.csv"
  submit_activation \
    "activation_prompt_jitter_${pct_label}_cfseg_full" \
    "$jitter_manifest" \
    "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prompt_jitter_${pct_label}" \
    "counterfactual_path" \
    "block3,block9,neck" \
    "whole,boundary,background" \
    "matched_cf_to_original,reverse_original_to_cf,nonmatched_cf_to_original" \
    "32"
done

for width in 16 64; do
  submit_activation \
    "activation_cfseg_bw${width}_matched_full" \
    "$BASE_MANIFEST" \
    "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw${width}" \
    "counterfactual_path" \
    "block3,block9,neck" \
    "whole,boundary,background,random_like_boundary,random_like_background" \
    "matched_cf_to_original" \
    "$width"
done

for width in 16 32 64; do
  submit_activation \
    "activation_cxrsd_bw${width}_random_matched_full" \
    "$PRISM_MANIFEST" \
    "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw${width}" \
    "prism_counterfactual_path" \
    "block3,block9,neck" \
    "whole,boundary,background,random_like_boundary,random_like_background" \
    "matched_cf_to_original" \
    "$width"
done

echo "Submitted missing-region robustness jobs."
echo "$job_log"
