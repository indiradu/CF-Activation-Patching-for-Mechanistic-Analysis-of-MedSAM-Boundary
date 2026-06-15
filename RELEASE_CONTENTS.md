# Release Contents

This repository is the clean code and aggregate-results package for the
counterfactual MedSAM activation-patching study. It is separate from the full
working experiment folder.

## Included

- `src/`: reusable mask metrics, manifest helpers, and spatial region utilities.
- `scripts/`: reproducibility scripts for MedSAM inference, activation patching,
  statistics, and result-figure/table generation.
- `tests/`: lightweight tests for manifest parsing, metrics, and region masks.
- `configs/`: example path configuration.
- `cluster/`: sanitized Slurm templates with placeholder/default paths.
- `reproducibility/results/`: aggregate CSVs that support the paper tables.
- `docs/`: general data-access and runbook documentation.

## Included Result Groups

- `generator_comparison/`: 50-case CF-Seg and CXR-SD generator comparison.
- `cfseg_activation_controls/`: 50-case CF-Seg activation-control summaries.
- `cxrsd_activation_controls/`: 50-case CXR-SD activation-control summaries.
- `nih_effusion_200/`: scaled 200-case CF-Seg MedSAM and activation-patching
  summaries.
- `nih_effusion_200_diffusion/`: scaled 200-case CXR-SD diffusion
  counterfactual summaries.
- `supervisor_robustness_neck/`: prompt-box jitter and boundary-width sweeps.
- `supervisor_interior_neck/`: interior/background control summaries.
- `failure_prediction/`: lightweight failure-monitoring summary.

## Excluded

- raw NIH ChestX-ray14 images;
- CheXMask masks;
- generated counterfactual images;
- MedSAM, CF-Seg, CXR-SD, PRISM, or RadEdit checkpoints;
- per-case predicted masks and activation patch masks;
- manuscript source, compiled PDFs, and qualitative CXR panels containing
  patient-derived pixels;
- historical run notes with local cluster paths;
- logs, cache files, and local environment folders.

## Release Sanity Checks

From this directory:

```bash
pytest
rg -n "YOUR_REAL_NAME|YOUR_USERNAME|PRIVATE_HOST|PRIVATE_ACCOUNT|PRIVATE_EMAIL" .
git status --short
```
