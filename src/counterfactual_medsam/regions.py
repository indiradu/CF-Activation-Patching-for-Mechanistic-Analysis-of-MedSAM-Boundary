"""Spatial regions for boundary-focused activation probing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L")) > 0


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def boundary_band(mask: np.ndarray, width: int = 5) -> np.ndarray:
    mask = mask.astype(bool)
    if width <= 0:
        raise ValueError("Boundary width must be positive")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(mask, structure=structure, iterations=width)
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=width)
    return np.logical_and(dilated, ~eroded)


def region_mask(mask: np.ndarray, region: str, boundary_width: int = 5) -> np.ndarray:
    mask = mask.astype(bool)
    region = region.lower()
    if region == "whole":
        return np.ones_like(mask, dtype=bool)
    if region == "boundary":
        return boundary_band(mask, width=boundary_width)
    if region == "interior":
        return np.logical_and(mask, ~boundary_band(mask, width=boundary_width))
    if region == "background":
        return ~mask
    raise ValueError(f"Unknown region: {region}")


def dilate(mask: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 0:
        raise ValueError("Dilation width must be positive")
    structure = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_dilation(mask.astype(bool), structure=structure, iterations=width)


def requires_prediction(region: str) -> bool:
    return region.lower() in {
        "error",
        "false_negative",
        "false_positive",
        "failure_boundary",
        "missed_boundary",
        "extra_boundary",
    }


def probing_region(
    reference: np.ndarray,
    region: str,
    boundary_width: int = 5,
    prediction: np.ndarray | None = None,
) -> np.ndarray:
    """Return a full-resolution patching region.

    Failure-aware regions use the original MedSAM prediction so patching targets
    the boundary area where the baseline segmentor actually disagreed with the
    reference, instead of every anatomical boundary token.
    """

    reference = reference.astype(bool)
    region = region.lower()
    if not requires_prediction(region):
        return region_mask(reference, region, boundary_width=boundary_width)
    if prediction is None:
        raise ValueError(f"Region `{region}` requires an original prediction mask")

    prediction = prediction.astype(bool)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction/reference shape mismatch: {prediction.shape} vs {reference.shape}")

    false_negative = np.logical_and(reference, ~prediction)
    false_positive = np.logical_and(prediction, ~reference)
    error = np.logical_or(false_negative, false_positive)
    boundary = boundary_band(reference, width=boundary_width)

    if region == "error":
        return error
    if region == "false_negative":
        return false_negative
    if region == "false_positive":
        return false_positive
    if region == "failure_boundary":
        return np.logical_and(boundary, dilate(error, width=boundary_width))
    if region == "missed_boundary":
        return np.logical_and(boundary, dilate(false_negative, width=boundary_width))
    if region == "extra_boundary":
        return np.logical_and(boundary, dilate(false_positive, width=boundary_width))
    raise ValueError(f"Unknown region: {region}")


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {values.shape} vs {mask.shape}")
    if not mask.any():
        return float("nan")
    return float(values[mask].mean())
