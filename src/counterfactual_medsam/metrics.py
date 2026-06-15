"""Mask metrics for the minimum viable MedSAM comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def read_binary_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L")) > 0


def dice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    denom = pred.sum() + ref.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, ref).sum() / denom)


def iou_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, ref).sum() / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def hd95(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred_surface = _surface(prediction)
    ref_surface = _surface(reference)
    if not pred_surface.any() and not ref_surface.any():
        return 0.0
    if not pred_surface.any() or not ref_surface.any():
        return float("inf")

    pred_to_ref = ndimage.distance_transform_edt(~ref_surface)[pred_surface]
    ref_to_pred = ndimage.distance_transform_edt(~pred_surface)[ref_surface]
    distances = np.concatenate([pred_to_ref, ref_to_pred])
    return float(np.percentile(distances, 95))


def boundary_f1(prediction: np.ndarray, reference: np.ndarray, tolerance: int = 3) -> float:
    pred_surface = _surface(prediction)
    ref_surface = _surface(reference)
    if not pred_surface.any() and not ref_surface.any():
        return 1.0
    if not pred_surface.any() or not ref_surface.any():
        return 0.0

    structure = ndimage.generate_binary_structure(2, 1)
    pred_dilated = ndimage.binary_dilation(pred_surface, structure=structure, iterations=tolerance)
    ref_dilated = ndimage.binary_dilation(ref_surface, structure=structure, iterations=tolerance)

    precision = np.logical_and(pred_surface, ref_dilated).sum() / max(pred_surface.sum(), 1)
    recall = np.logical_and(ref_surface, pred_dilated).sum() / max(ref_surface.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def area_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    ref_area = reference.astype(bool).sum()
    if ref_area == 0:
        return 0.0 if prediction.astype(bool).sum() == 0 else float("inf")
    return float((prediction.astype(bool).sum() - ref_area) / ref_area)


def evaluate_mask_pair(
    *,
    prediction_path: Path,
    reference_path: Path,
    boundary_tolerance: int = 3,
) -> dict[str, float]:
    prediction = read_binary_mask(prediction_path)
    reference = read_binary_mask(reference_path)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for {prediction_path} and {reference_path}: "
            f"{prediction.shape} vs {reference.shape}"
        )
    return {
        "dice": dice_score(prediction, reference),
        "iou": iou_score(prediction, reference),
        "boundary_f1": boundary_f1(prediction, reference, tolerance=boundary_tolerance),
        "hd95": hd95(prediction, reference),
        "area_error": area_error(prediction, reference),
    }
