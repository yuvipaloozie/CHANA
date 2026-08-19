import numpy as np

from chana.metrics import binary_dice, binary_iou, centroid_match_scores, hd95


def test_identical_masks_have_perfect_pixel_metrics():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 9:21] = 1
    assert binary_iou(mask, mask) == 1.0
    assert binary_dice(mask, mask) == 1.0
    assert hd95(mask, mask) == 0.0


def test_centroid_matching_uses_distance_gate():
    reference = np.array([[10, 10], [50, 50]], dtype=float)
    prediction = np.array([[12, 11], [90, 90], [53, 51]], dtype=float)
    score = centroid_match_scores(reference, prediction, max_distance_px=5)
    assert score.true_positive == 2
    assert score.false_positive == 1
    assert score.false_negative == 0
    assert np.isclose(score.f1, 0.8)


def test_one_empty_mask_has_undefined_hd95():
    assert np.isnan(hd95(np.zeros((8, 8)), np.ones((8, 8))))
