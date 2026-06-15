# NIH Effusion 200-Case Scale Run

This note records the scale-up from the original 50-case MICCAI workshop experiment to a 200-case NIH ChestX-ray14 pleural-effusion cohort.

## Cohort

- Dataset: NIH ChestX-ray14 from the Kaggle mirror.
- Cohort: first 200 AP/PA images with pleural-effusion labels selected by `scripts/build_nih_manifest.py`.
- Mask source: CheXMask ChestX-Ray8 original-resolution RLE masks, decoded with `scripts/export_chexmask_nih_masks.py`.
- Prompt: fixed MedSAM lung bounding box derived from the CheXMask combined lung mask with 12-pixel margin.

## Completed Outputs

- `data/manifests/nih_effusion_200.csv`
- `data/manifests/nih_effusion_200_with_masks.csv`
- `data/manifests/nih_effusion_200_with_masks_boxes.csv`
- `data/manifests/nih_effusion_200_runtime.csv`
- `data/manifests/nih_effusion_200_with_counterfactuals.csv`
- `data/manifests/nih_effusion_200_activation_manifest.csv`
- `outputs/medsam/nih_effusion_200/original/metrics_vs_chexmask.csv`
- `outputs/medsam/nih_effusion_200/counterfactual_cfseg_pe0/metrics_vs_chexmask.csv`
- `outputs/activation_probe/nih_effusion_200_neck_controls_bw32/patching_metrics.csv`
- `outputs/activation_probe/nih_effusion_200_neck_controls_bw32/summary_by_control_region.csv`
- `reproducibility/results/nih_effusion_200/condition_means.csv`
- `reproducibility/results/nih_effusion_200/key_scaled_results.csv`

## Segmentation Metrics

| Condition | n | Dice | IoU | Boundary F1 | HD95 | Area error |
|---|---:|---:|---:|---:|---:|---:|
| Original | 200 | 0.725775 | 0.587169 | 0.156144 | 123.486481 | 0.324384 |
| CF-Seg counterfactual | 200 | 0.850389 | 0.753977 | 0.217208 | 80.038554 | 0.178125 |

## Activation Patching

The first single 200-case activation job was cancelled because it was slower than necessary. The manifest was split into four 50-case chunks and submitted as parallel neck-layer jobs:

- chunk 00: Slurm job `142220`
- chunk 01: Slurm job `142221`
- chunk 02: Slurm job `142222`
- chunk 03: Slurm job `142223`

Each chunk runs `matched_cf_to_original`, `reverse_original_to_cf`, and `nonmatched_cf_to_original` controls at the encoder neck over `whole`, `boundary`, `background`, and `interior` regions with a 32-pixel boundary band.

All four chunks completed successfully. Their `patching_metrics.csv` files were concatenated into:

- `outputs/activation_probe/nih_effusion_200_neck_controls_bw32/patching_metrics.csv`

The combined table contains 2,400 patched-mask rows across 200 cases. It was summarized with:

```bash
PYTHONPATH=src python scripts/summarize_activation_patching.py \
  --patching-metrics outputs/activation_probe/nih_effusion_200_neck_controls_bw32/patching_metrics.csv \
  --output-dir outputs/activation_probe/nih_effusion_200_neck_controls_bw32
```

Key neck-layer activation-patching results:

| Condition | n | Dice | IoU | Boundary F1 | HD95 | Area error |
|---|---:|---:|---:|---:|---:|---:|
| Original | 200 | 0.725775 | 0.587169 | 0.156144 | 123.486481 | 0.324384 |
| CF-Seg counterfactual | 200 | 0.850389 | 0.753977 | 0.217208 | 80.038554 | 0.178125 |
| Matched whole | 200 | 0.850389 | 0.753977 | 0.217208 | 80.038554 | 0.178125 |
| Matched background/context | 200 | 0.824050 | 0.721935 | 0.251295 | 86.348427 | 0.190388 |
| Matched boundary | 200 | 0.761063 | 0.635257 | 0.163905 | 107.296562 | 0.239776 |
| Matched interior | 200 | 0.787270 | 0.661787 | 0.173905 | 110.493856 | 0.329501 |
| Reverse whole | 200 | 0.725775 | 0.587169 | 0.156144 | 123.486481 | 0.324384 |
| Non-matched whole | 200 | 0.662216 | 0.507397 | 0.057151 | 128.298821 | 0.176584 |

Interpretation: the 200-case scaled run preserves the 50-case mechanism. Matched whole-map counterfactual patching reproduces the CF-Seg counterfactual behavior, reverse whole-map patching returns to original-like behavior, and non-matched donor patching performs substantially worse. At the encoder neck, background/context patching gives the strongest Boundary F1 signal, above boundary and interior patching.
