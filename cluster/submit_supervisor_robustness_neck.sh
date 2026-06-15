#!/usr/bin/env bash
# Submit focused neck-layer robustness experiments for prompt jitter and region controls.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MANIFEST="${BASE_MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_activation_manifest.csv}"
PRISM_MANIFEST="${PRISM_MANIFEST:-$PROJECT_DIR/outputs/medsam/nih_effusion_50/prism_diffusion_v2/manifest_with_predictions.csv}"

cd "$PROJECT_DIR"
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
job_log="$PROJECT_DIR/logs/supervisor_robustness_neck_jobs_${timestamp}.txt"
touch "$job_log"

submit_activation() {
  local label="$1"
  local manifest="$2"
  local output_dir="$3"
  local counterfactual_column="$4"
  local regions="$5"
  local controls="$6"
  local boundary_width="$7"
  local job_id

  job_id="$(
    PROJECT_DIR="$PROJECT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    MANIFEST="$manifest" \
    OUTPUT_DIR="$output_dir" \
    COUNTERFACTUAL_COLUMN="$counterfactual_column" \
    LAYERS="neck" \
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
    "neck_prompt_jitter_${pct_label}_cfseg" \
    "$jitter_manifest" \
    "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prompt_jitter_${pct_label}_neck" \
    "counterfactual_path" \
    "whole,boundary,background" \
    "matched_cf_to_original,reverse_original_to_cf,nonmatched_cf_to_original" \
    "32"
done

for width in 16 32 64; do
  submit_activation \
    "neck_cfseg_bw${width}_matched" \
    "$BASE_MANIFEST" \
    "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw${width}_neck" \
    "counterfactual_path" \
    "whole,boundary,background,random_like_boundary,random_like_background" \
    "matched_cf_to_original" \
    "$width"
done

for width in 16 32 64; do
  submit_activation \
    "neck_cxrsd_bw${width}_matched" \
    "$PRISM_MANIFEST" \
    "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw${width}_neck" \
    "prism_counterfactual_path" \
    "whole,boundary,background,random_like_boundary,random_like_background" \
    "matched_cf_to_original" \
    "$width"
done

echo "Submitted neck-layer robustness jobs."
echo "$job_log"
