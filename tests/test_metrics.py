import numpy as np

from counterfactual_medsam.metrics import boundary_f1, dice_score, hd95, iou_score


def test_perfect_overlap_metrics() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0
    assert boundary_f1(mask, mask, tolerance=1) == 1.0
    assert hd95(mask, mask) == 0.0


def test_partial_overlap_metrics() -> None:
    ref = np.zeros((8, 8), dtype=bool)
    pred = np.zeros((8, 8), dtype=bool)
    ref[2:6, 2:6] = True
    pred[3:7, 2:6] = True

    assert 0.0 < dice_score(pred, ref) < 1.0
    assert 0.0 < iou_score(pred, ref) < 1.0
    assert hd95(pred, ref) > 0.0
