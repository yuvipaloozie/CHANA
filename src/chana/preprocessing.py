"""Deterministic image preprocessing used by the CHANA V9 pipeline."""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_preprocessed_rgb(
    image: np.ndarray,
    size: tuple[int, int] = (512, 512),
    input_order: str = "bgr",
) -> np.ndarray:
    """Normalize an already V9-preprocessed RGB image for model input."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if input_order not in {"bgr", "rgb"}:
        raise ValueError("input_order must be 'bgr' or 'rgb'")
    image_u8 = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    rgb = cv2.cvtColor(image_u8, cv2.COLOR_BGR2RGB) if input_order == "bgr" else image_u8
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    scaled = rgb.astype(np.float32) / 255.0
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_v9(
    image: np.ndarray,
    size: tuple[int, int] = (512, 512),
    input_order: str = "bgr",
) -> np.ndarray:
    """Apply the V9 LAB/CLAHE/top-hat and ImageNet-normalization pipeline.

    Parameters
    ----------
    image:
        Three-channel uint8-like image.
    size:
        Output ``(width, height)``. The historical experiments used 512 × 512.
    input_order:
        ``"bgr"`` for an OpenCV-loaded image or ``"rgb"`` otherwise.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if input_order not in {"bgr", "rgb"}:
        raise ValueError("input_order must be 'bgr' or 'rgb'")

    image_u8 = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    rgb = cv2.cvtColor(image_u8, cv2.COLOR_BGR2RGB) if input_order == "bgr" else image_u8
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness_clahe = clahe.apply(lightness)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    nuclei_map = cv2.morphologyEx(lightness_clahe, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.addWeighted(lightness_clahe, 1.0, nuclei_map, 0.8, 0.0)
    processed_rgb = cv2.cvtColor(
        cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB
    )

    scaled = processed_rgb.astype(np.float32) / 255.0
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD
