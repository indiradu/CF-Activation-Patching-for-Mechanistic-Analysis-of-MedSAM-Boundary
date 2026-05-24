# Week 1 Runbook

Goal: produce the first 20-50 matched original/counterfactual MedSAM comparisons, with clean manifests and no sensitive data in Git.

If MIMIC-CXR-JPG access is slow, use the NIH ChestX-ray14 + CheXMask route in [dataset_alternatives.md](dataset_alternatives.md).

## 0. Local Repository Rule

Keep the GitHub repo private for now. Commit code and configs only. Never commit:

- MIMIC-CXR images
- CheXMask masks
- CF-Seg counterfactual images
- MedSAM or CF-Seg checkpoints
- cluster scratch paths with protected data

## 1. Confirm Data Access

You need credentialed access to:

- MIMIC-CXR-JPG v2.1.0
- MIMIC-CXR labels/metadata CSVs
- CheXMask masks or a mask index that maps `dicom_id` to a lung mask path
- CF-Seg expert subset if available under the license/terms

## 2. Clone External Repositories On The GPU Machine

Run this on the university GPU machine, from the project root:

```bash
bash scripts/setup_external_repos.sh external
```

This creates:

```text
external/MedSAM
external/CF-Seg
external/SAM-Med2D
external/PRISM
```

Only MedSAM and CF-Seg are needed for the first milestone. SAM-Med2D and PRISM are optional later.

## 3. Create Local Paths Config

Copy the example config:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

Edit `configs/paths.local.yaml` on the GPU machine so it points to your dataset, checkpoint, and output roots.

Then validate paths:

```bash
python scripts/check_paths.py --config configs/paths.local.yaml
```

It is okay if optional paths are missing early. The important first paths are MIMIC-CXR-JPG, a mask index/source, MedSAM repo, and MedSAM checkpoint.

## 4. Build A Pleural-Effusion Manifest

Example:

```bash
python scripts/build_manifest.py \
  --mimic-root /path/to/mimic-cxr-jpg/2.1.0 \
  --metadata-csv /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --labels-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
  --mask-index /path/to/mask_index.csv \
  --output data/manifests/mimic_pe_50.csv \
  --limit 50
```

The mask index is optional for the first pass, but required before metric evaluation. It should contain either `dicom_id` or `image_id`, plus `mask_path`.

No-MIMIC alternative:

```bash
python -m pip install kagglehub
python scripts/download_nih_chestxray14.py \
  --cache-dir /scratch/$USER/kagglehub

# Use the printed "Path to dataset files" as NIH_ROOT below.
python scripts/build_nih_manifest.py \
  --image-root $NIH_ROOT \
  --labels-csv $NIH_ROOT/Data_Entry_2017.csv \
  --mask-index /path/to/chexmask_chestxray8_mask_index.csv \
  --output data/manifests/nih_effusion_50.csv \
  --limit 50
```

For CheXMask, do not download the full ZIP if the NIH MVP is the target. Download only the ChestX-ray8 mask CSV:

```bash
bash scripts/download_chexmask_subset.sh /scratch/$USER/miccai_workshop/chexmask
```

This fetches `OriginalResolution/ChestX-Ray8.csv` from CheXMask v1.0.0 plus the small README/license/checksum files.

## 5. Add Prompt Boxes

For the controlled first experiment, derive one lung bounding box from the reference lung mask and reuse the same box for original and counterfactual images:

```bash
python scripts/add_boxes_from_masks.py \
  --manifest data/manifests/mimic_pe_50.csv \
  --output data/manifests/mimic_pe_50_with_boxes.csv \
  --margin 12
```

Using mask-derived boxes is acceptable for the mechanistic MVP because we are isolating MedSAM boundary behavior under matched prompts. Later, add automatic boxes as a robustness check.

## 6. Run MedSAM On Original Images

```bash
python scripts/run_medsam_cli_manifest.py \
  --manifest data/manifests/mimic_pe_50_with_boxes.csv \
  --medsam-repo external/MedSAM \
  --checkpoint checkpoints/medsam_vit_b.pth \
  --image-column image_path \
  --prediction-column medsam_original_mask_path \
  --output-dir outputs/medsam/original
```

The wrapper assumes the official MedSAM CLI entrypoint `MedSAM_Inference.py`. If the upstream CLI changes, update only this wrapper.

## 7. Generate Or Load Counterfactuals

Start with CF-Seg because it is the closest prior and directly targets pseudo-healthy pleural-effusion CXR lung segmentation. Once counterfactual files exist, add their paths into `counterfactual_path` in the manifest.

For the NIH MVP, first convert the runtime manifest into CF-Seg's expected 256x256 causal-gen layout:

```bash
python scripts/prepare_cfseg_nih_inputs.py \
  --manifest data/manifests/nih_effusion_50_runtime.csv \
  --output-root data/runtime/cfseg_nih_effusion_50
```

Then download/extract CF-Seg checkpoint bundles into `checkpoints/cfseg/` and run:

```bash
sbatch cluster/run_cfseg_generate_nih_effusion.slurm
```

After generation, import the pseudo-healthy images back into the original-resolution manifest:

```bash
python scripts/import_cfseg_counterfactuals.py \
  --manifest data/manifests/nih_effusion_50_runtime.csv \
  --cfseg-dir outputs/counterfactuals/cfseg_raw/p/nih_effusion_50_pe_to_no_finding/test \
  --output-root outputs/counterfactuals/cfseg_nih_effusion_50/original_resolution \
  --output-manifest data/manifests/nih_effusion_50_with_counterfactuals.csv
```

Note: CF-Seg was released around a MIMIC-style data layout. The NIH adapter is an MVP engineering bridge, not a final domain-validity claim. Treat its generated images as a way to debug the MedSAM causal-probing pipeline unless visual QA confirms clinically plausible edits.

For the first paper table, every case should have:

```text
image_path
counterfactual_path
lung_mask_path
box_xyxy
```

## 8. Run MedSAM On Counterfactuals

```bash
python scripts/run_medsam_cli_manifest.py \
  --manifest data/manifests/mimic_pe_50_with_counterfactuals.csv \
  --medsam-repo external/MedSAM \
  --checkpoint checkpoints/medsam_vit_b.pth \
  --image-column counterfactual_path \
  --prediction-column medsam_counterfactual_mask_path \
  --output-dir outputs/medsam/counterfactual
```

## 9. Compare Masks

```bash
python scripts/evaluate_masks.py \
  --manifest data/manifests/mimic_pe_50_with_counterfactuals.csv \
  --prediction-column medsam_original_mask_path \
  --reference-column lung_mask_path \
  --output outputs/tables/medsam_original_metrics.csv

python scripts/evaluate_masks.py \
  --manifest data/manifests/mimic_pe_50_with_counterfactuals.csv \
  --prediction-column medsam_counterfactual_mask_path \
  --reference-column lung_mask_path \
  --output outputs/tables/medsam_counterfactual_metrics.csv
```

First sanity plot/table:

| Run | Dice | IoU | Boundary F1 | HD95 |
| --- | --- | --- | --- | --- |
| MedSAM original | lower | lower | lower | higher |
| MedSAM counterfactual | higher | higher | higher | lower |

If this trend appears on 20-50 cases, move to activation hooks.

## 10. Activation Summaries And Patching

Once `counterfactual_path` is populated, run the first mechanistic probe:

```bash
sbatch cluster/run_activation_probe_nih_effusion.slurm
```

The script measures original/counterfactual activation differences at `block3`, `block6`, `block9`, and `neck`. With `RUN_PATCHING=1`, it also patches matched counterfactual activations into original-image inference for whole-map, boundary, interior, and background regions.
