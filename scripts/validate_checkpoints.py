#!/usr/bin/env python
"""Verify local CHANA checkpoint files against the public model registry."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from chana.checkpoints import (
    load_checkpoint_registry,
    resolve_checkpoint_path,
    verify_checkpoint_file,
)
from chana.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "manifests" / "model_registry.csv",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        help="validate only this model ID; repeat to select more than one",
    )
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="verify file size and SHA-256 without importing TensorFlow models",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registry = load_checkpoint_registry(args.registry)
    if args.model_id:
        requested = set(args.model_id)
        unknown = requested - set(registry)
        if unknown:
            raise SystemExit(f"unknown model IDs: {sorted(unknown)}")
        specs = [registry[model_id] for model_id in registry if model_id in requested]
    else:
        specs = list(registry.values())

    for spec in specs:
        try:
            path = resolve_checkpoint_path(spec, args.weights_dir)
            verify_checkpoint_file(spec, path)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from error

        message = f"{spec.model_id}: size/hash verified ({path.name})"
        if not args.hash_only:
            model = build_model(spec.architecture, encoder_weights=None)
            model.load_weights(path)
            message += f", architecture load verified ({model.count_params()} parameters)"
            del model
            try:
                import tensorflow as tf

                tf.keras.backend.clear_session()
            finally:
                gc.collect()
        print(message)


if __name__ == "__main__":
    main()
