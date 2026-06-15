# Counterfactual Activation Patching for MedSAM

This repository contains code and aggregate results for a MI4MedFM 2026
workshop paper on mechanistic probing of MedSAM boundary failures with
counterfactual chest X-rays.

Working title:
**Counterfactual Activation Patching as a Mechanistic Probe of Boundary
Failures in MedSAM**.

## Main Claim

On NIH ChestX-ray14 pleural-effusion cases, patient-matched pseudo-healthy
counterfactuals change MedSAM internal image-encoder representations in a way
that causally affects lung-boundary segmentation. Matched counterfactual
activation patching restores counterfactual-like segmentation behavior, reverse
patching restores original-like failure, and non-matched donors do not reproduce
the effect. Region controls suggest that broader context/background
representations can carry more corrective signal than a thin boundary band
alone, so the boundary failure is interpreted as context-mediated rather than
purely local edge loss.

## Repository Layout

- `src/counterfactual_medsam/`: reusable project code.
- `scripts/`: command-line entrypoints for manifests, MedSAM inference,
  activation patching, statistics, and result-figure/table generation.
- `tests/`: lightweight unit tests.
- `configs/`: path template examples.
- `cluster/`: sanitized Slurm templates.
- `reproducibility/`: aggregate CSV outputs used to verify paper results.
- `docs/`: data access notes and high-level runbook.

## Included Results

- 50-case CF-Seg and CXR-SD generator comparison summaries.
- 200-case NIH ChestX-ray14 pleural-effusion CF-Seg activation-patching
  summaries.
- 200-case CXR-SD diffusion counterfactual summaries.
- Supervisor-requested robustness checks for prompt-box jitter, boundary-band
  width sweeps, and interior/background control comparisons.
- Failure-prediction summary tables using output and activation-derived
  features.

## Data Policy

This release intentionally does not include raw chest X-rays, CheXMask masks,
generated counterfactual images, predicted masks, patch masks, or model
checkpoints. These files must be restored locally according to the licenses and
access rules of NIH ChestX-ray14, CheXMask, MedSAM, CF-Seg, and the diffusion
generator used in the paper.

The manuscript source, compiled PDF, and qualitative CXR figure panels are
intentionally not included here so the paper can be revised independently during
review and patient-derived pixels are not duplicated in the public code release.
This repository is the code and aggregate-results package.

## Quick Start

Install the lightweight package and run tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

Optional experiment dependencies:

```bash
python -m pip install -e ".[experiments,generators,data]"
```

The main aggregate results are already included under
`reproducibility/results/`.

## Paper

The paper is submitted separately. After review, this repository can be linked
from the manuscript as the public implementation and aggregate-results release.
