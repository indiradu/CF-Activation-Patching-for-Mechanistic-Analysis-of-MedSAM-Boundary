# Data And Model Access Checklist

## Highest Priority

| Item | Why we need it | Action |
| --- | --- | --- |
| MIMIC-CXR-JPG v2.1.0 | Main CXR source with pleural-effusion and no-finding labels | Confirm PhysioNet credentialed access, CITI training, and DUA |
| CheXMask | Silver lung masks and filtering signal | Download PhysioNet CSV/RLE masks after access is available |
| CF-Seg expert MIMIC subset | Stronger mask evaluation for diseased cases | Download from CF-Seg links if allowed by the terms |
| MedSAM checkpoint | Target model for mechanistic probing | Download official checkpoint to local/cluster `checkpoints/` |
| CF-Seg checkpoints | First counterfactual generator and U-Net baseline | Download causal-gen and CheXMask-Seg checkpoints |

## Secondary Priority

| Item | Why we need it | When |
| --- | --- | --- |
| SAM-Med2D weights | Strong medical SAM-family baseline | After MedSAM patching MVP |
| PRISM code/weights | Diffusion counterfactual generator comparison | Only if CF-Seg path works |
| PadChest + CheXMask | External validation | After MIMIC result |
| nnU-Net | Strong supervised baseline | Optional, if enough train/test data and time |

## Access Notes

- MIMIC-CXR-JPG is a credentialed PhysioNet resource. Only users who complete training, sign the DUA, and are credentialed can access the files.
- CheXMask includes masks but not the original CXR images; original image access still depends on the source dataset.
- CF-Seg provides code, checkpoints, dataset preparation scripts, and links to selected expert lung segmentations. Check the license and data terms before redistributing anything.
- Model checkpoints should stay outside Git. Use `configs/paths.example.yaml` as the template for local paths.

## First Manifest Columns

Create a CSV or Parquet manifest with these columns:

```text
case_id
subject_id
study_id
dicom_id
split
view_position
has_pleural_effusion
has_no_finding
image_path
lung_mask_path
mask_source
counterfactual_path
counterfactual_source
box_xyxy
notes
```

Keep patient identifiers limited to dataset-internal IDs required for reproducibility. Do not include names, dates, free-text reports, or any PHI.
