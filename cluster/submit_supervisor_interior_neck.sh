#!/usr/bin/env bash
# Submit focused neck-layer lung-interior controls requested in supervisor feedback.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MANIFEST="${BASE_MANIFEST:-$PROJECT_DIR/data/manifests/nih_effusion_50_activation_manifest.csv}"
CXRSD_MANIFEST="${CXRSD_MANIFEST:-$PROJECT_DIR/outputs/medsam/nih_effusion_50/prism_diffusion_v2/manifest_with_predictions.csv}"

cd "$PROJECT_DIR"
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
job_log="$PROJECT_DIR/logs/supervisor_interior_neck_jobs_${timestamp}.txt"
touch "$job_log"

submit_activation() {
  local label="$1"
  local manifest="$2"
  local output_dir="$3"
  local counterfactual_column="$4"
  local job_id

  job_id="$(
    PROJECT_DIR="$PROJECT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    MANIFEST="$manifest" \
    OUTPUT_DIR="$output_dir" \
    COUNTERFACTUAL_COLUMN="$counterfactual_column" \
    LAYERS="neck" \
    REGIONS="interior,random_like_interior" \
    CONTROLS="matched_cf_to_original" \
    BOUNDARY_WIDTH="32" \
    sbatch --parsable --export=ALL cluster/run_activation_probe_nih_effusion.slurm
  )"
  printf "%s,%s\n" "$label" "$job_id" | tee -a "$job_log"
}

submit_activation \
  "neck_cfseg_interior_bw32" \
  "$BASE_MANIFEST" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_interior_bw32_neck" \
  "counterfactual_path"

submit_activation \
  "neck_cxrsd_interior_bw32" \
  "$CXRSD_MANIFEST" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_interior_bw32_neck" \
  "prism_counterfactual_path"

echo "Submitted interior neck jobs."
echo "$job_log"
