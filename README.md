# Counterfactual Activation Patching for MedSAM

This repository contains code and aggregate results for a MI4MedFM 2026
workshop paper on mechanistic probing of MedSAM boundary failures with
counterfactual chest X-rays.

Working title:
**Counterfactual Activation Patching Reveals Context-Mediated Boundary Failures
in MedSAM**.

## Main Claim

On NIH ChestX-ray14 pleural-effusion cases, patient-matched pseudo-healthy
counterfactuals change MedSAM internal image-encoder representations in a way
that causally affects lung-boundary segmentation. Matched counterfactual
activation patching restores counterfactual-like segmentation behavior, reverse
patching restores original-like failure, and non-matched donors do not reproduce
the effect. Region controls suggest that broader context/background
representations carry more corrective signal than a thin boundary band alone.

## Repository Layout

- `src/counterfactual_medsam/`: reusable project code.
- `scripts/`: command-line entrypoints for manifests, MedSAM inference,
  activation patching, statistics, and paper assets.
- `tests/`: lightweight unit tests.
- `configs/`: path template examples.
- `cluster/`: sanitized Slurm templates.
- `paper/`: LNCS source, result tables, and non-patient figures.
- `reproducibility/`: aggregate CSV outputs used to verify paper results.
- `docs/`: data access notes and high-level runbook.

## Data Policy

This release intentionally does not include raw chest X-rays, CheXMask masks,
generated counterfactual images, predicted masks, patch masks, or model
checkpoints. These files must be restored locally according to the licenses and
access rules of NIH ChestX-ray14, CheXMask, MedSAM, CF-Seg, and the diffusion
generator used in the paper.

The paper source contains a fallback placeholder for the qualitative CXR panel
because that panel contains patient-derived pixels and is not part of this
public code package.

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

The LNCS source is in `paper/main.tex`. To compile it, place Springer's
`llncs.cls` in `paper/` or install it in your TeX path, then run your preferred
LaTeX build command from `paper/`.

The paper currently states that implementation and aggregate outputs will be
released publicly after review.
