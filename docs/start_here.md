# Start Here

## Decision

Yes: create a GitHub repository, but make it **private** at first. The repo should contain code, configs, documentation, and reproducibility scripts only. It should not contain MIMIC-CXR images, CheXMask masks, CF-Seg generated counterfactuals, MedSAM checkpoints, or university cluster paths with sensitive details.

This folder has been initialized as a local Git repository so it does not accidentally use the parent home-directory Git repo.

## Minimum Viable Paper

The first paper-worthy result is a causal intervention table, not a giant benchmark:

| Condition | Expected result |
| --- | --- |
| MedSAM on original pleural-effusion CXR | Boundary failure / lower Dice / higher HD95 |
| MedSAM on matched pseudo-healthy counterfactual | Improved boundary quality |
| Original inference with matched counterfactual boundary activations patched in | Improves boundary metrics |
| Counterfactual inference with diseased boundary activations patched in | Worsens boundary metrics |
| Random/non-matched/interior/background patching | Smaller or no improvement |

If this table works on 20-50 cases, the workshop story is alive.

## Week 1 Plan

1. Confirm access to MIMIC-CXR-JPG and CheXMask.
2. Clone MedSAM and CF-Seg into `external/` on the GPU machine.
3. Download MedSAM and CF-Seg checkpoints to `checkpoints/` or cluster scratch.
4. Build a manifest CSV with `image_id`, `patient_id`, `study_id`, `view`, `label`, `image_path`, `mask_path`, and later `counterfactual_path`.
5. Run MedSAM inference on 20-50 original CXR images with fixed bounding-box prompts.
6. Run or load CF-Seg pseudo-healthy counterfactuals for the same 20-50 cases.
7. Compare MedSAM masks on original vs counterfactual images.

## Week 2 Plan

1. Add MedSAM hooks for image encoder blocks 3, 6, 9, final neck, and mask decoder tokens.
2. Save activation summaries, not full activations by default.
3. Implement whole-map activation patching for one layer.
4. Implement boundary-band, interior, and background spatial patching.
5. Run matched vs non-matched controls.

## Week 3 Plan

1. Scale to 20-50 cases.
2. Add Boundary F1, HD95, Dice, IoU, and lung area/volume error.
3. Generate layer-by-layer patching heatmaps.
4. Draft the intro, related work, and methods while experiments are still running.

## Week 4 Plan

1. Add SAM-Med2D or CF-Seg U-Net baseline, but only after the patching result is stable.
2. Add failure prediction with activation features.
3. Prepare figures for the MI4MedFM submission.
4. Decide whether PRISM is feasible as an additional generator comparison.
