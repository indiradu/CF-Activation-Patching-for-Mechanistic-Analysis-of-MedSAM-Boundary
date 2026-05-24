# Scripts

Add command-line entrypoints here once the data paths and GPU environment are confirmed.

Initial entrypoints:

- `build_manifest.py`: collect eligible MIMIC-CXR/CheXMask cases.
- `build_nih_manifest.py`: collect eligible NIH ChestX-ray14/CheXMask cases when MIMIC access is slow.
- `download_nih_chestxray14.py`: download NIH ChestX-ray14 with KaggleHub into a chosen cache directory.
- `download_chexmask_subset.py`: download only the CheXMask files needed for NIH/ChestX-ray14.
- `download_chexmask_subset.sh`: wget-based CheXMask NIH subset downloader, usually faster on clusters.
- `export_chexmask_nih_masks.py`: decode CheXMask RLE lung masks for selected NIH manifest rows.
- `validate_manifest_assets.py`: sanity-check image/mask paths, dimensions, and prompt boxes.
- `make_manifest_qa_sheet.py`: render image/mask/box overlays for visual QA.
- `run_medsam_batch_manifest.py`: run MedSAM once over a manifest and save binary predictions.
- `run_medsam_inference.py`: run MedSAM on original and counterfactual images.
- `add_boxes_from_masks.py`: derive controlled MedSAM prompt boxes from reference masks.
- `run_medsam_cli_manifest.py`: run the official MedSAM CLI over manifest rows.
- `evaluate_masks.py`: compute Dice, IoU, Boundary F1, HD95, and lung area error.
- `materialize_manifest_assets.py`: copy manifest images/masks into a portable runtime directory and rewrite paths.
- `prepare_cfseg_nih_inputs.py`: resize NIH images/masks into the CF-Seg causal-gen layout.
- `download_cfseg_checkpoints.py`: download CF-Seg checkpoint bundles from the release Google Drive links.
- `import_cfseg_counterfactuals.py`: resize CF-Seg counterfactuals back to original image size and add paths to a manifest.
- `run_medsam_activation_probe.py`: compute matched original/counterfactual activation summaries and optional activation patching masks.

Week 2 entrypoints:

- `run_medsam_activation_probe.py`: performs the first activation cache/summary and matched patching experiments.
