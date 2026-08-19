"""Validation for a redistributable CHANA inference example."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy import ndimage as ndi

from .postprocessing import separate_objects


class SampleValidationError(ValueError):
    """Raised when a sample-data manifest or file is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SampleValidationError(f"OpenCV could not read {path}")
    return image


def _validate_file(root: Path, name: str, specification: dict[str, Any]) -> np.ndarray:
    path = root / specification["path"]
    if not path.is_file():
        raise SampleValidationError(f"missing {name}: {path}")
    observed_hash = _sha256(path)
    if observed_hash != specification["sha256"]:
        raise SampleValidationError(
            f"SHA-256 mismatch for {name}: {observed_hash}"
        )

    image = _load_image(path)
    expected_shape = (
        specification["height"],
        specification["width"],
    )
    if specification["channels"] > 1:
        expected_shape += (specification["channels"],)
    if image.shape != expected_shape:
        raise SampleValidationError(
            f"shape mismatch for {name}: {image.shape} != {expected_shape}"
        )
    if str(image.dtype) != specification["dtype"]:
        raise SampleValidationError(
            f"dtype mismatch for {name}: {image.dtype} != {specification['dtype']}"
        )
    return image


def validate_sample_manifest(
    manifest_path: Path,
    *,
    require_cleared: bool = False,
    require_checkpoint_linked: bool = False,
) -> dict[str, Any]:
    """Validate sample files, hashes, dimensions, alignment, and mask encoding."""
    manifest_path = Path(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SampleValidationError("unsupported or missing schema_version")

    clearance = manifest.get("redistribution", {}).get("status")
    if require_cleared and clearance != "cleared":
        raise SampleValidationError(
            f"redistribution status is {clearance!r}, not 'cleared'"
        )
    checkpoint_model_id = manifest.get("overlay", {}).get("checkpoint_model_id")
    if require_checkpoint_linked and checkpoint_model_id in {None, "", "unresolved"}:
        raise SampleValidationError(
            "exact originating checkpoint is unresolved"
        )

    root = manifest_path.parent
    arrays = {
        name: _validate_file(root, name, specification)
        for name, specification in manifest["files"].items()
    }
    spatial_shapes = {array.shape[:2] for array in arrays.values()}
    if len(spatial_shapes) != 1:
        raise SampleValidationError(f"sample files are not aligned: {spatial_shapes}")

    mask = arrays["reference_mask"]
    mask_specification = manifest["reference_mask"]
    expected_values = {
        mask_specification["background_value"],
        mask_specification["foreground_value"],
    }
    observed_values = set(np.unique(mask).tolist())
    if observed_values != expected_values:
        raise SampleValidationError(
            f"mask values mismatch: {observed_values} != {expected_values}"
        )

    foreground = mask == mask_specification["foreground_value"]
    foreground_pixels = int(foreground.sum())
    foreground_fraction = float(foreground.mean())
    if foreground_pixels != mask_specification["foreground_pixels"]:
        raise SampleValidationError(
            "reference-mask foreground-pixel count does not match the manifest"
        )
    if not np.isclose(
        foreground_fraction,
        mask_specification["foreground_fraction"],
        rtol=0,
        atol=1e-15,
    ):
        raise SampleValidationError(
            "reference-mask foreground fraction does not match the manifest"
        )

    output_specification = manifest["expected_output"]
    probability = arrays["probability_map"]
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise SampleValidationError("probability map must contain finite values in [0, 1]")
    threshold = output_specification["probability_threshold"]
    if output_specification["threshold_operator"] == "greater_than_or_equal":
        calculated_prediction = probability >= threshold
    elif output_specification["threshold_operator"] == "greater_than":
        calculated_prediction = probability > threshold
    else:
        raise SampleValidationError("unsupported probability threshold operator")
    if output_specification["fill_holes"]:
        calculated_prediction = ndi.binary_fill_holes(calculated_prediction)

    binary_prediction = arrays["binary_prediction"]
    binary_values = set(np.unique(binary_prediction).tolist())
    if binary_values != {0, 255}:
        raise SampleValidationError(
            f"binary prediction values mismatch: {binary_values} != {{0, 255}}"
        )
    deposited_prediction = binary_prediction > 0
    if not np.array_equal(deposited_prediction, calculated_prediction):
        raise SampleValidationError(
            "binary prediction does not match the thresholded probability map"
        )
    predicted_pixels = int(deposited_prediction.sum())
    predicted_fraction = float(deposited_prediction.mean())
    if predicted_pixels != output_specification["predicted_foreground_pixels"]:
        raise SampleValidationError(
            "predicted foreground-pixel count does not match the manifest"
        )
    if not np.isclose(
        predicted_fraction,
        output_specification["predicted_foreground_fraction"],
        rtol=0,
        atol=1e-15,
    ):
        raise SampleValidationError(
            "predicted foreground fraction does not match the manifest"
        )

    deposited_labels = arrays["watershed_labels"].astype(np.int32)
    if not np.array_equal(deposited_labels > 0, deposited_prediction):
        raise SampleValidationError(
            "watershed-label foreground does not match the binary prediction"
        )
    deposited_label_values = set(np.unique(deposited_labels).tolist()) - {0}
    if len(deposited_label_values) != output_specification["watershed_label_count"]:
        raise SampleValidationError(
            "watershed label count does not match the manifest"
        )
    calculated_labels = separate_objects(
        deposited_prediction,
        min_peak_distance=output_specification["watershed_peak_distance"],
    )
    calculated_label_values = set(np.unique(calculated_labels).tolist()) - {0}
    mapping: dict[int, int] = {}
    for deposited_label in deposited_label_values:
        matched = set(
            np.unique(calculated_labels[deposited_labels == deposited_label]).tolist()
        ) - {0}
        if len(matched) != 1:
            raise SampleValidationError(
                "current watershed partition differs from the deposited labels"
            )
        mapping[deposited_label] = matched.pop()
    if set(mapping.values()) != calculated_label_values:
        raise SampleValidationError(
            "current watershed partition differs from the deposited labels"
        )

    reference_edges = cv2.morphologyEx(
        foreground.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    prediction_edges = cv2.morphologyEx(
        deposited_prediction.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    expected_overlay = arrays["input_image"].copy()
    colors = manifest["overlay"]["colors_rgb"]
    expected_overlay[reference_edges] = colors["expert_reference_mask"][::-1]
    expected_overlay[prediction_edges] = colors["model_prediction"][::-1]
    if not np.array_equal(
        arrays["reference_prediction_overlay"], expected_overlay
    ):
        raise SampleValidationError(
            "overlay does not match the deposited masks and configured colors"
        )

    return {
        "sample_id": manifest["sample_id"],
        "redistribution_status": clearance,
        "shape": next(iter(spatial_shapes)),
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": foreground_fraction,
        "predicted_foreground_pixels": predicted_pixels,
        "predicted_foreground_fraction": predicted_fraction,
        "watershed_label_count": len(deposited_label_values),
        "internal_consistency_status": output_specification[
            "internal_consistency_status"
        ],
        "overlay_verification_status": manifest["overlay"]["verification_status"],
        "overlay_checkpoint_model_id": checkpoint_model_id,
    }
