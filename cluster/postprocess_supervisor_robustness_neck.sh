#!/usr/bin/env bash
# Summarize focused neck-layer robustness experiments.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

for pct_label in 05 10 15; do
  output_dir="$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prompt_jitter_${pct_label}_neck"
  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_activation_patching.py \
    --patching-metrics "$output_dir/patching_metrics.csv" \
    --output-dir "$output_dir"
done

for output_dir in \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw16_neck" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw32_neck" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_controls_bw64_neck" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw16_neck" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw32_neck" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_random_bw64_neck"; do
  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_activation_patching.py \
    --patching-metrics "$output_dir/patching_metrics.csv" \
    --output-dir "$output_dir"
done

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_supervisor_robustness.py \
  --project-root "$PROJECT_DIR" \
  --output-dir "$PROJECT_DIR/outputs/supervisor_robustness_neck" \
  --activation-suffix _neck
