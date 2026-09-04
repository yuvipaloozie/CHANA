#!/usr/bin/env python
"""Run CHANA inference on one microscopy image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chana.checkpoints import resolve_and_verify_checkpoint
from chana.inference import predict_tiled
from chana.models import build_checkpoint_model
from chana.postprocessing import filter_and_measure, measurements_frame, separate_objects


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--architecture", choices=["unet", "unetpp", "transunet"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model-id",
        help="semantic model ID from manifests/model_registry.csv",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="directory containing a canonical or preserved legacy checkpoint file",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "manifests" / "model_registry.csv",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument("--max-area", type=float, default=10_000.0)
    parser.add_argument("--min-peak-distance", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.model_id:
        if args.architecture or args.checkpoint:
            raise ValueError(
                "--model-id cannot be combined with --architecture or --checkpoint"
            )
        if args.weights_dir is None:
            raise ValueError("--weights-dir is required with --model-id")
        spec, checkpoint = resolve_and_verify_checkpoint(
            args.registry, args.model_id, args.weights_dir
        )
        architecture = spec.architecture
    else:
        if args.architecture is None or args.checkpoint is None:
            raise ValueError(
                "provide --model-id/--weights-dir or --architecture/--checkpoint"
            )
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        architecture = args.architecture
        checkpoint = args.checkpoint

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {args.input}")

    model = build_checkpoint_model(architecture)
    model.load_weights(str(checkpoint))
    probability = predict_tiled(
        model, image, architecture, batch_size=args.batch_size
    )
    binary = probability > args.threshold
    labels = separate_objects(binary, min_peak_distance=args.min_peak_distance)
    labels, measurements = filter_and_measure(labels, args.min_area, args.max_area)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    np.save(args.output_dir / f"{stem}_probability.npy", probability.astype(np.float32))
    cv2.imwrite(str(args.output_dir / f"{stem}_mask.png"), binary.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output_dir / f"{stem}_labels.tif"), labels.astype(np.uint16))
    measurements_frame(measurements).to_csv(
        args.output_dir / f"{stem}_objects.csv", index=False
    )
    print(f"Detected {len(measurements)} filtered objects; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
