"""Architecture-aware prediction and tiled-image inference."""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

from .preprocessing import preprocess_v9


def select_probability_output(prediction: Any, architecture: str) -> np.ndarray:
    """Select the full-resolution output from a model prediction."""
    if isinstance(prediction, (list, tuple)):
        if architecture.lower() in {"unetpp", "unet++"}:
            prediction = prediction[-1]
        elif architecture.lower() == "transunet":
            prediction = prediction[0]
        else:
            prediction = prediction[0]
    array = np.asarray(prediction)
    if array.ndim == 4:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"expected batch prediction with 3 dimensions, got {array.shape}")
    return array


def predict_tiled(
    model: Any,
    image_bgr: np.ndarray,
    architecture: str,
    tile_size: int = 512,
    batch_size: int = 8,
    preprocessor: Callable[..., np.ndarray] = preprocess_v9,
) -> np.ndarray:
    """Predict a large OpenCV image by non-overlapping padded tiles."""
    height, width = image_bgr.shape[:2]
    padded_height = math.ceil(height / tile_size) * tile_size
    padded_width = math.ceil(width / tile_size) * tile_size
    padded = cv2.copyMakeBorder(
        image_bgr,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    tiles: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for top in range(0, padded_height, tile_size):
        for left in range(0, padded_width, tile_size):
            tile = padded[top : top + tile_size, left : left + tile_size]
            tiles.append(preprocessor(tile, size=(tile_size, tile_size), input_order="bgr"))
            positions.append((top, left))

    probability = np.zeros((padded_height, padded_width), dtype=np.float32)
    for start in range(0, len(tiles), batch_size):
        batch = np.stack(tiles[start : start + batch_size])
        output = model.predict(batch, verbose=0)
        selected = select_probability_output(output, architecture)
        for tile_probability, (top, left) in zip(selected, positions[start : start + batch_size]):
            probability[top : top + tile_size, left : left + tile_size] = tile_probability
    return probability[:height, :width]
