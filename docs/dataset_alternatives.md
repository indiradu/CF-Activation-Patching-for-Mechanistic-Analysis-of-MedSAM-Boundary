# Dataset Alternatives If MIMIC Access Is Slow

MIMIC-CXR-JPG is still a strong final-paper dataset, but it requires credentialing, training, and a data use agreement. We can start without it.

## Recommended First Substitute: NIH ChestX-ray14 + CheXMask

Use this if we want to keep the **pleural effusion lung-boundary** story.

Why it fits:

- NIH ChestX-ray14 has 112,120 frontal chest X-rays with labels including `Effusion` and `No Finding`.
- CheXMask includes anatomical segmentation masks for ChestX-ray8/NIH images.
- This preserves the core design: diseased effusion CXR, no-finding controls, lung masks, MedSAM boundary behavior.
- Access is usually much lighter than MIMIC; use the official NIH/Kaggle terms and do not redistribute images.

Limitations:

- Labels are text-mined and noisier than expert labels.
- CheXMask masks are silver masks, not manual expert masks.
- CF-Seg was designed around MIMIC/PadChest pleural effusion, so generated counterfactual quality must be checked manually.

Best use:

- Use NIH + CheXMask for the first 20-50 case MVP.
- If MIMIC access arrives later, use MIMIC as the stronger validation set.

## Fast Pipeline Debug Dataset: Mendeley Lung Segmentation

Use this if we only want to test MedSAM inference, metrics, prompt boxes, and plotting quickly.

Why it fits:

- It contains 6,810 chest X-rays with binary lung masks from Darwin, Montgomery, and Shenzhen.
- It is easy to use for lung segmentation pipeline debugging.

Limitations:

- It does not directly support the pleural-effusion counterfactual story.
- It is not enough for the main mechanistic novelty claim by itself.

Best use:

- Dry-run MedSAM wrappers and metrics before the real disease/counterfactual experiment.

## Disease-Specific Pivot: SIIM-ACR Pneumothorax

Use this only if we are willing to pivot away from pleural effusion.

Why it fits:

- It has chest X-rays and pixel-level pneumothorax masks.
- It is good for boundary failure and lesion segmentation experiments.

Limitations:

- It changes the target from lung-boundary failure under pleural effusion to pneumothorax lesion/boundary failure.
- CF-Seg pseudo-healthy effusion generation is no longer the right generator.
- We would need a different counterfactual strategy.

## Lower-Priority Options

| Dataset | Usefulness | Main issue |
| --- | --- | --- |
| CheXpert + CheXMask | Large CXR source with effusion/no-finding labels | Requires registration/user agreement; labels and masks are still weak/silver |
| PadChest + CheXMask | Strong external validation for CF-Seg-style story | Access/data cleaning can still be slow |
| VinDr-CXR + CheXMask | Radiologist annotations and pleural-effusion labels | PhysioNet/DUA-style access may not solve the bottleneck |
| JSRT/SCR | Clean small lung masks | Too small and not disease-counterfactual enough |
| Montgomery/Shenzhen | Public lung masks/TB examples | Good sanity check, weak fit to pleural-effusion boundary failure |

## Current Recommendation

Start with:

```text
NIH ChestX-ray14 images
+ NIH labels CSV
+ CheXMask ChestX-ray8 masks
+ MedSAM
+ CF-Seg or PRISM counterfactual generation later
```

This lets us keep the workshop contribution stable: **counterfactual activation patching for MedSAM boundary-failure mechanisms**.

