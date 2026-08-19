"""Pixel, boundary, and centroid-matched object evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops


def binary_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    union = np.logical_or(reference, prediction).sum()
    return 1.0 if union == 0 else float(np.logical_and(reference, prediction).sum() / union)


def binary_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    denominator = reference.sum() + prediction.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(reference, prediction).sum() / denominator)


def binary_average_precision(reference: np.ndarray, probability: np.ndarray) -> float:
    """Average precision for a binary mask without an sklearn dependency.

    Returns NaN when the reference contains no foreground pixels, matching the
    paper evaluation. Stable descending sorting and threshold grouping mirror
    ``sklearn.metrics.average_precision_score`` for finite binary inputs.
    """
    reference = np.asarray(reference, dtype=np.uint8).ravel()
    probability = np.asarray(probability, dtype=np.float64).ravel()
    if reference.shape != probability.shape:
        raise ValueError("reference and probability must contain the same number of pixels")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    if not reference.any():
        return float("nan")

    order = np.argsort(probability, kind="mergesort")[::-1]
    truth_sorted = reference[order]
    probability_sorted = probability[order]
    distinct = np.where(np.diff(probability_sorted))[0]
    threshold_indices = np.r_[distinct, truth_sorted.size - 1]
    true_positive = np.cumsum(truth_sorted, dtype=np.float64)[threshold_indices]
    false_positive = 1 + threshold_indices - true_positive
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / true_positive[-1]
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def hd95(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance in pixels.

    Returns NaN when exactly one mask is empty and 0 when both are empty.
    """
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if not reference.any() and not prediction.any():
        return 0.0
    if not reference.any() or not prediction.any():
        return float("nan")
    reference_boundary = reference ^ ndi.binary_erosion(reference)
    prediction_boundary = prediction ^ ndi.binary_erosion(prediction)
    to_prediction = ndi.distance_transform_edt(~prediction_boundary)[reference_boundary]
    to_reference = ndi.distance_transform_edt(~reference_boundary)[prediction_boundary]
    return float(np.percentile(np.concatenate((to_prediction, to_reference)), 95))


def component_centroids(binary_mask: np.ndarray) -> np.ndarray:
    regions = regionprops(label(np.asarray(binary_mask, dtype=bool)))
    if not regions:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([region.centroid for region in regions], dtype=np.float64)


@dataclass(frozen=True)
class ObjectScores:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def centroid_match_scores(
    reference_centroids: np.ndarray,
    predicted_centroids: np.ndarray,
    max_distance_px: float = 25.0,
) -> ObjectScores:
    """Match objects one-to-one with Hungarian assignment and a distance gate."""
    reference = np.asarray(reference_centroids, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(predicted_centroids, dtype=np.float64).reshape(-1, 2)
    if len(reference) == 0 and len(predicted) == 0:
        return ObjectScores(0, 0, 0, 1.0, 1.0, 1.0)
    if len(reference) == 0 or len(predicted) == 0:
        true_positive = 0
    else:
        row_indices, column_indices = linear_sum_assignment(cdist(reference, predicted))
        distances = cdist(reference, predicted)[row_indices, column_indices]
        true_positive = int(np.count_nonzero(distances <= max_distance_px))
    false_positive = int(len(predicted) - true_positive)
    false_negative = int(len(reference) - true_positive)
    precision = true_positive / (true_positive + false_positive) if len(predicted) else 0.0
    recall = true_positive / (true_positive + false_negative) if len(reference) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return ObjectScores(true_positive, false_positive, false_negative, precision, recall, f1)
