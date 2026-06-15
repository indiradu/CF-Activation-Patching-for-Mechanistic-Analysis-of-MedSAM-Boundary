# Reproducibility Package

This folder contains the aggregate numerical outputs needed to verify the
paper tables without redistributing protected or patient-derived image data.

## Included Results

- `results/generator_comparison/`
  - Mean MedSAM metrics for original, CF-Seg, and CXR-SD images.
  - Paired bootstrap confidence intervals and Wilcoxon tests.
- `results/cfseg_activation_controls/`
  - Key CF-Seg activation-patching control table.
  - Paired statistics for matched, reverse, non-matched, and region controls.
- `results/cxrsd_activation_controls/`
  - Key CXR-SD activation-patching control table.
  - Paired statistics for the diffusion generator robustness check.
- `results/failure_prediction/`
  - Exploratory failure-monitoring summary.

## Excluded Data

The repository intentionally excludes:

- raw NIH ChestX-ray14 images;
- CheXMask mask files;
- generated counterfactual images;
- MedSAM, CF-Seg, Stable Diffusion, or RadEdit checkpoints;
- per-case predicted masks and patch-mask PNGs;
- local cluster logs and machine-specific absolute paths.

These exclusions are required for a clean anonymous repository and avoid
redistributing third-party data or patient-derived images.

## Reproducing Tables

After restoring the required datasets/checkpoints locally and creating the
runtime manifests, the main tables can be regenerated with:

```bash
python scripts/bootstrap_statistical_tables.py \
  --original-metrics outputs/medsam/nih_effusion_50/original/metrics_vs_chexmask.csv \
  --counterfactual-metrics outputs/medsam/nih_effusion_50/counterfactual_cfseg_pe0/metrics_vs_chexmask.csv \
  --patching-metrics outputs/activation_probe/nih_effusion_50_controls_bw32/patching_metrics.csv \
  --output-dir outputs/statistics/nih_effusion_50_cfseg_controls_bw32
```

Paper assets and compact LaTeX tables can be regenerated from completed CSVs
with:

```bash
python scripts/make_paper_assets.py
```

## Anonymous Review Note

For double-blind review, use an anonymous repository URL and avoid adding
personal GitHub usernames, institutional cluster hostnames, or absolute local
paths to manifests, logs, or documentation.
