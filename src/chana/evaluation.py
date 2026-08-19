"""Evaluation of one probability map against an expert semantic mask."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .metrics import (
    binary_average_precision,
    binary_dice,
    binary_iou,
    centroid_match_scores,
    hd95,
)
from .postprocessing import filter_and_measure, separate_objects


def evaluate_probability_map(
    reference_mask: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    min_peak_distance: int = 20,
    min_area: float = 50.0,
    max_area: float = 10_000.0,
    max_match_distance: float = 25.0,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Return manuscript-style pixel, boundary, object, and count metrics."""
    reference = np.asarray(reference_mask) > 0
    probability = np.asarray(probability, dtype=np.float32)
    if reference.shape != probability.shape:
        raise ValueError(
            f"reference and probability shapes differ: {reference.shape} != {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")

    prediction = ndi.binary_fill_holes(probability > threshold)
    boundary_distance = hd95(reference, prediction)
    if not reference.any() and not prediction.any():
        hd95_status = "both_empty"
    elif not reference.any():
        hd95_status = "empty_reference"
    elif not prediction.any():
        hd95_status = "empty_prediction"
    elif np.isfinite(boundary_distance):
        hd95_status = "finite"
    else:
        hd95_status = "compute_failed"

    reference_labels = separate_objects(reference, min_peak_distance)
    prediction_labels = separate_objects(prediction, min_peak_distance)
    _, reference_objects = filter_and_measure(reference_labels, min_area, max_area)
    _, prediction_objects = filter_and_measure(prediction_labels, min_area, max_area)
    reference_centroids = np.asarray(
        [(item.centroid_y, item.centroid_x) for item in reference_objects],
        dtype=np.float64,
    ).reshape(-1, 2)
    prediction_centroids = np.asarray(
        [(item.centroid_y, item.centroid_x) for item in prediction_objects],
        dtype=np.float64,
    ).reshape(-1, 2)
    objects = centroid_match_scores(
        reference_centroids,
        prediction_centroids,
        max_distance_px=max_match_distance,
    )

    signed_count_error = len(prediction_objects) - len(reference_objects)
    metrics: dict[str, float | int | str] = {
        "iou": binary_iou(reference, prediction),
        "dice": binary_dice(reference, prediction),
        "average_precision": binary_average_precision(reference, probability),
        "hd95_px": boundary_distance,
        "hd95_status": hd95_status,
        "reference_objects": len(reference_objects),
        "predicted_objects": len(prediction_objects),
        "object_true_positive": objects.true_positive,
        "object_false_positive": objects.false_positive,
        "object_false_negative": objects.false_negative,
        "object_precision": objects.precision,
        "object_recall": objects.recall,
        "object_f1": objects.f1,
        "count_absolute_error": abs(signed_count_error),
        "count_signed_error": signed_count_error,
    }
    return metrics, prediction.astype(np.uint8)
