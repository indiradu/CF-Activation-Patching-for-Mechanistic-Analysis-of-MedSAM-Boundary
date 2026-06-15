#!/usr/bin/env bash
# Summarize focused neck-layer lung-interior controls.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

for output_dir in \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_interior_bw32_neck" \
  "$PROJECT_DIR/outputs/activation_probe/nih_effusion_50_prism_v2_interior_bw32_neck"; do
  PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" scripts/summarize_activation_patching.py \
    --patching-metrics "$output_dir/patching_metrics.csv" \
    --output-dir "$output_dir"
done

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd

project = Path(".").resolve()
specs = [
    ("CF-Seg", project / "outputs/activation_probe/nih_effusion_50_interior_bw32_neck/patching_metrics.csv"),
    ("CXR-SD", project / "outputs/activation_probe/nih_effusion_50_prism_v2_interior_bw32_neck/patching_metrics.csv"),
]
rows = []
for source, path in specs:
    df = pd.read_csv(path)
    df = df[df["control"].eq("matched_cf_to_original")]
    for region, group in df.groupby("region"):
        rows.append({
            "source": source,
            "region": region,
            "dice": group["dice"].mean(),
            "iou": group["iou"].mean(),
            "boundary_f1": group["boundary_f1"].mean(),
            "hd95": group["hd95"].mean(),
            "area_error": group["area_error"].mean(),
        })
out_dir = project / "outputs/supervisor_interior_neck"
out_dir.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame(rows).sort_values(["source", "region"])
summary.to_csv(out_dir / "interior_summary.csv", index=False)
print(summary.to_string(index=False))
PY
