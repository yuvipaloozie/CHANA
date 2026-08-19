#!/usr/bin/env python
"""Compare registered CHANA checkpoints on paired images and masks."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from chana.checkpoints import resolve_and_verify_checkpoint
from chana.evaluation import evaluate_probability_map
from chana.inference import predict_tiled
from chana.preprocessing import normalize_preprocessed_rgb, preprocess_v9


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_IDS = ["unetpp_baseline", "unetpp_curriculum"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-csv",
        required=True,
        type=Path,
        help="CSV with image_id, image_path, and mask_path; relative paths use the CSV directory",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="repeatable semantic model ID; defaults to U-Net++ baseline and curriculum",
    )
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "manifests" / "model_registry.csv"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-stage",
        choices=["raw", "preprocessed"],
        default="raw",
        help="raw applies V9 enhancement; preprocessed applies only ImageNet normalization",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument("--max-area", type=float, default=10_000.0)
    parser.add_argument("--min-peak-distance", type=int, default=20)
    parser.add_argument("--max-match-distance", type=float, default=25.0)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_pairs(path: Path) -> list[dict[str, str | Path]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "image_path", "mask_path"}
    if not rows:
        raise ValueError(f"{path} contains no image-mask pairs")
    if not required <= set(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    pairs: list[dict[str, str | Path]] = []
    seen: set[str] = set()
    for row in rows:
        image_id = row["image_id"].strip()
        if not image_id or image_id in seen:
            raise ValueError(f"blank or duplicate image_id: {image_id!r}")
        seen.add(image_id)
        image_path = Path(row["image_path"].strip())
        mask_path = Path(row["mask_path"].strip())
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not mask_path.is_absolute():
            mask_path = path.parent / mask_path
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"missing image or mask for {image_id}: {image_path}, {mask_path}"
            )
        pairs.append(
            {"image_id": image_id, "image_path": image_path, "mask_path": mask_path}
        )
    return pairs


def main():
    args = parse_args()
    model_ids = args.model_ids or DEFAULT_MODEL_IDS
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("--model-id values must be unique")
    pairs = load_pairs(args.pairs_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = preprocess_v9 if args.input_stage == "raw" else normalize_preprocessed_rgb

    # TensorFlow is optional for repository validation and required only for inference.
    from chana.models import build_model
    import tensorflow as tf

    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        spec, checkpoint = resolve_and_verify_checkpoint(
            args.registry, model_id, args.weights_dir
        )
        model = build_model(spec.architecture, encoder_weights=None)
        model.load_weights(str(checkpoint))

        for pair in pairs:
            image_path = Path(pair["image_path"])
            mask_path = Path(pair["mask_path"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            reference = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or reference is None:
                raise ValueError(f"OpenCV could not read {image_path} or {mask_path}")
            if reference.ndim > 2:
                reference = reference[..., 0]
            if reference.shape != image.shape[:2]:
                raise ValueError(
                    f"image and mask shapes differ for {pair['image_id']}: "
                    f"{image.shape[:2]} != {reference.shape}"
                )

            probability = predict_tiled(
                model,
                image,
                spec.architecture,
                batch_size=args.batch_size,
                preprocessor=preprocessor,
            )
            metrics, prediction = evaluate_probability_map(
                reference,
                probability,
                threshold=args.threshold,
                min_peak_distance=args.min_peak_distance,
                min_area=args.min_area,
                max_area=args.max_area,
                max_match_distance=args.max_match_distance,
            )
            rows.append(
                {
                    "image_id": pair["image_id"],
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "model_id": model_id,
                    "architecture": spec.architecture,
                    "canonical_checkpoint_file": spec.checkpoint_file,
                    "resolved_checkpoint_file": checkpoint.name,
                    "checkpoint_sha256": spec.sha256,
                    "input_stage": args.input_stage,
                    "threshold": args.threshold,
                    **metrics,
                }
            )

            if args.save_predictions:
                model_dir = args.output_dir / model_id
                model_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    model_dir / f"{pair['image_id']}_probability.npy",
                    probability.astype(np.float32),
                )
                cv2.imwrite(
                    str(model_dir / f"{pair['image_id']}_mask.png"),
                    prediction * 255,
                )
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    frame = pd.DataFrame(rows)
    metrics_path = args.output_dir / "per_image_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    summary_columns = [
        "iou",
        "dice",
        "average_precision",
        "hd95_px",
        "object_precision",
        "object_recall",
        "object_f1",
        "count_absolute_error",
        "count_signed_error",
    ]
    summary = frame.groupby("model_id", as_index=False)[summary_columns].mean()
    summary_path = args.output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {metrics_path} and {summary_path}")


if __name__ == "__main__":
    main()
