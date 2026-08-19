#!/usr/bin/env python
"""Validate the public-example candidate and its provenance manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from chana.sample_validation import SampleValidationError, validate_sample_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("sample_data/public_example/manifest.yml"),
    )
    parser.add_argument(
        "--require-cleared",
        action="store_true",
        help="also require confirmed public-redistribution clearance",
    )
    parser.add_argument(
        "--require-checkpoint-linked",
        action="store_true",
        help="also require an exact originating checkpoint model ID",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = validate_sample_manifest(
            args.manifest,
            require_cleared=args.require_cleared,
            require_checkpoint_linked=args.require_checkpoint_linked,
        )
    except SampleValidationError as exc:
        raise SystemExit(f"Sample-data validation failed: {exc}") from exc
    print(
        f"Validated {report['sample_id']}: {report['shape'][1]}x{report['shape'][0]}, "
        f"foreground={report['foreground_pixels']} pixels "
        f"({report['foreground_fraction']:.6%}), "
        f"prediction={report['predicted_foreground_pixels']} pixels, "
        f"watershed={report['watershed_label_count']} labels, "
        f"redistribution={report['redistribution_status']}, "
        f"internal_consistency={report['internal_consistency_status']}, "
        f"checkpoint={report['overlay_checkpoint_model_id']}"
    )


if __name__ == "__main__":
    main()
