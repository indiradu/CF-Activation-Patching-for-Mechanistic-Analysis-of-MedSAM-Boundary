# Release Contents

This folder is the clean package intended for a future public GitHub release
after review. It is separate from the working experiment folder.

## Included

- `src/`: reusable mask metrics, manifest helpers, and spatial region utilities.
- `scripts/`: reproducibility scripts for MedSAM inference, activation patching,
  statistics, and result-figure/table generation.
- `tests/`: lightweight tests for manifest parsing, metrics, and region masks.
- `configs/`: example path configuration.
- `cluster/`: sanitized Slurm templates with placeholder/default paths.
- `reproducibility/results/`: aggregate CSVs that support the paper tables.
- `docs/`: general data-access and runbook documentation.

## Excluded

- raw NIH ChestX-ray14 images;
- CheXMask masks;
- generated counterfactual images;
- MedSAM, CF-Seg, CXR-SD, PRISM, or RadEdit checkpoints;
- per-case predicted masks and activation patch masks;
- manuscript source and qualitative CXR panels containing patient-derived pixels;
- historical run notes with local cluster paths;
- logs, cache files, and local environment folders.

## Before Public Push

From this directory:

```bash
pytest
rg -n "YOUR_REAL_NAME|YOUR_USERNAME|PRIVATE_HOST|PRIVATE_ACCOUNT|PRIVATE_EMAIL" .
git status --short
```

Then initialize and push:

```bash
git init
git add .
git commit -m "Public reproducibility release"
git branch -M main
git remote add origin https://github.com/<your-account>/<repo-name>.git
git push -u origin main
```
