#!/usr/bin/env bash
# Evaluate and summarize supervisor robustness experiments after Slurm jobs finish.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

for pct_label in 05 10 15; do
  original_dir="$PROJECT_DIR/outputs/medsam/nih_effusion_50/prompt_jitter_${pct_label}/original"
  cfseg_dir="$PROJECT_DIR/outputs/medsam/nih_effusion_50/prompt_jitter_${pct_label}/cfseg"

  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/evaluate_masks.py \
    --manifest "$original_dir/manifest_with_predictions.csv" \
    --prediction-column medsam_original_jitter_mask_path \
    --output "$original_dir/metrics_vs_chexmask.csv"

  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/evaluate_masks.py \
    --manifest "$cfseg_dir/manifest_with_predictions.csv" \
    --prediction-column medsam_cfseg_jitter_mask_path \
    --output "$cfseg_dir/metrics_vs_chexmask.csv"

  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_activation_patching.py \
    --patching-metrics "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prompt_jitter_${pct_label}/patching_metrics.csv" \
    --output-dir "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prompt_jitter_${pct_label}"
done

for output_dir in \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw16" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw64" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw16" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw32" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw64"; do
  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_activation_patching.py \
    --patching-metrics "$output_dir/patching_metrics.csv" \
    --output-dir "$output_dir"
done

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_supervisor_robustness.py \
  --project-root "$PROJECT_DIR" \
  --output-dir "$PROJECT_DIR/outputs/supervisor_robustness"
