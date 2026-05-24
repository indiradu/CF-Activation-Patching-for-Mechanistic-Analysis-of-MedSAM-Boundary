import numpy as np

from counterfactual_medsam.regions import boundary_band, probing_region, region_mask, requires_prediction, resize_mask


def test_boundary_band_marks_edges_without_filling_interior() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True

    band = boundary_band(mask, width=1)

    assert band[2, 4]
    assert band[4, 2]
    assert not band[4, 4]


def test_region_masks_partition_main_regions() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True

    interior = region_mask(mask, "interior", boundary_width=1)
    background = region_mask(mask, "background", boundary_width=1)

    assert interior[4, 4]
    assert not interior[2, 2]
    assert background[0, 0]
    assert not background[4, 4]


def test_resize_mask_uses_nearest_neighbor() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    resized = resize_mask(mask, (8, 8))

    assert resized.dtype == bool
    assert resized.sum() == 16


def test_failure_boundary_uses_prediction_errors_near_reference_edge() -> None:
    reference = np.zeros((9, 9), dtype=bool)
    reference[2:7, 2:7] = True
    prediction = reference.copy()
    prediction[5:7, 2:7] = False

    missed = probing_region(reference, "missed_boundary", boundary_width=1, prediction=prediction)

    assert requires_prediction("missed_boundary")
    assert missed[6, 4]
    assert not missed[2, 4]


def test_prediction_region_requires_prediction_mask() -> None:
    reference = np.zeros((5, 5), dtype=bool)

    try:
        probing_region(reference, "failure_boundary", boundary_width=1)
    except ValueError as exc:
        assert "requires an original prediction" in str(exc)
    else:
        raise AssertionError("Expected missing prediction to raise")
