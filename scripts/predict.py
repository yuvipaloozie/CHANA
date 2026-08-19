#!/usr/bin/env python
"""Run CHANA inference on one microscopy image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chana.inference import predict_tiled
from chana.models import build_model
from chana.postprocessing import filter_and_measure, measurements_frame, separate_objects


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=["unet", "unetpp", "transunet"])
    parser.add_argument("--checkpoint", required=True, type=Path)
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
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {args.input}")

    model = build_model(args.architecture, encoder_weights=None)
    model.load_weights(str(args.checkpoint))
    probability = predict_tiled(
        model, image, args.architecture, batch_size=args.batch_size
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
