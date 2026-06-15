# Expert Review Protocol for Counterfactual MedSAM Sanity Check

Purpose: collect a small blinded plausibility review without presenting it as clinical ground truth. This addresses the supervisor recommendation to strengthen clinical grounding while keeping the paper's main claim mechanistic.

## Suggested Case Selection

Review 12 cases: 6 where CF-Seg most improves Boundary F1 and 6 where CF-Seg worsens or weakly improves Boundary F1. This gives both success and failure examples.

Selected success cases:

- `nih_00000023_002`
- `nih_00000061_012`
- `nih_00000041_006`
- `nih_00000056_001`
- `nih_00000061_002`
- `nih_00000061_017`

Selected failure/weak cases:

- `nih_00000061_007`
- `nih_00000032_032`
- `nih_00000012_000`
- `nih_00000032_001`
- `nih_00000032_035`
- `nih_00000013_004`

## Blinded Review Setup

For each case, show the original radiograph and 4-6 anonymized mask overlays in random order:

- Original MedSAM prediction
- MedSAM on CF-Seg counterfactual
- Matched activation-patched prediction
- Reverse-patched prediction
- Non-matched donor prediction
- Optional boundary/context regional patch predictions

Do not label methods during review. Use labels such as `Mask A`, `Mask B`, etc., and reveal the mapping only after scores are collected.

## Reviewer Questions

Use 1-5 Likert scores unless free-text is requested.

- Lower-lung boundary plausibility: does the lung boundary look anatomically plausible?
- Anatomy preservation: does the counterfactual preserve patient-specific thoracic anatomy outside the effusion-dominant region?
- Artifact severity: are there visible generator or mask artifacts?
- Preferred mask: which overlay is most plausible?
- Failure notes: what looks clinically wrong or ambiguous?

## Reporting Language

If this is performed by a radiologist or experienced annotator, report it as a small blinded plausibility review. If no expert is available, keep it as a planned sanity check and do not claim clinical validation.

Safe wording:

> We use weak/silver labels for quantitative evaluation and include a blinded qualitative plausibility review as a clinical sanity check; this does not constitute expert ground-truth annotation.
