"""Convert semantic masks into separated analytical objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


@dataclass(frozen=True)
class ObjectMeasurement:
    label: int
    area_px: float
    centroid_y: float
    centroid_x: float
    eccentricity: float
    perimeter_px: float
    circularity: float


def separate_objects(binary_mask: np.ndarray, min_peak_distance: int = 20) -> np.ndarray:
    """Separate touching foreground regions with distance-transform watershed."""
    foreground = np.asarray(binary_mask, dtype=bool)
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.int32)

    foreground = ndi.binary_fill_holes(foreground)
    distance = ndi.distance_transform_edt(foreground)
    coordinates = peak_local_max(
        distance,
        min_distance=min_peak_distance,
        labels=foreground.astype(np.uint8),
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if len(coordinates):
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    else:
        markers, _ = ndi.label(foreground)
    return watershed(-distance, markers, mask=foreground).astype(np.int32)


def filter_and_measure(
    labels: np.ndarray,
    min_area: float = 50.0,
    max_area: float = 10_000.0,
) -> tuple[np.ndarray, list[ObjectMeasurement]]:
    """Filter watershed objects by area and return relabeled measurements."""
    filtered = np.zeros(np.asarray(labels).shape, dtype=np.int32)
    measurements: list[ObjectMeasurement] = []
    next_label = 1
    for region in regionprops(np.asarray(labels, dtype=np.int32)):
        if not (min_area <= region.area <= max_area):
            continue
        filtered[labels == region.label] = next_label
        perimeter = float(region.perimeter)
        circularity = 0.0 if perimeter == 0 else float(4 * np.pi * region.area / perimeter**2)
        measurements.append(
            ObjectMeasurement(
                label=next_label,
                area_px=float(region.area),
                centroid_y=float(region.centroid[0]),
                centroid_x=float(region.centroid[1]),
                eccentricity=float(region.eccentricity),
                perimeter_px=perimeter,
                circularity=circularity,
            )
        )
        next_label += 1
    return filtered, measurements


def measurements_frame(measurements: list[ObjectMeasurement]) -> pd.DataFrame:
    columns = [field for field in ObjectMeasurement.__dataclass_fields__]
    return pd.DataFrame([asdict(item) for item in measurements], columns=columns)
