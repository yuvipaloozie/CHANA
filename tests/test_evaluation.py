import numpy as np

from chana.evaluation import evaluate_probability_map


def test_probability_evaluation_reports_perfect_pixel_and_object_metrics():
    reference = np.zeros((64, 64), dtype=np.uint8)
    reference[12:36, 14:38] = 1
    probability = reference.astype(np.float32)

    metrics, prediction = evaluate_probability_map(reference, probability)

    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["hd95_px"] == 0.0
    assert metrics["hd95_status"] == "finite"
    assert metrics["reference_objects"] == metrics["predicted_objects"] == 1
    assert metrics["object_f1"] == 1.0
    assert metrics["count_absolute_error"] == 0
    assert np.array_equal(prediction, reference)


def test_probability_threshold_is_strictly_greater_than_half():
    reference = np.zeros((16, 16), dtype=np.uint8)
    probability = np.full((16, 16), 0.5, dtype=np.float32)

    metrics, prediction = evaluate_probability_map(reference, probability)

    assert not prediction.any()
    assert metrics["iou"] == 1.0
    assert metrics["hd95_status"] == "both_empty"


def test_probability_evaluation_rejects_shape_mismatch():
    reference = np.zeros((16, 16), dtype=np.uint8)
    probability = np.zeros((8, 8), dtype=np.float32)

    try:
        evaluate_probability_map(reference, probability)
    except ValueError as error:
        assert "shapes differ" in str(error)
    else:
        raise AssertionError("shape mismatch was not rejected")
