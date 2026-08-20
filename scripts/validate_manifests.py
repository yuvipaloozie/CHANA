#!/usr/bin/env python
"""Validate the canonical CHANA model registry."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"
REGISTRY_COLUMNS = {
    "model_id",
    "architecture",
    "training_regime",
    "checkpoint_file",
    "legacy_checkpoint_file",
    "bytes",
    "sha256",
    "verified_environment",
}
PRIMARY_MODEL_IDS = {
    "unet_baseline",
    "unet_curriculum",
    "unetpp_baseline",
    "unetpp_curriculum",
    "transunet_baseline",
    "transunet_curriculum",
}


def _blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def validate(require_populated: bool = False, manifest_dir: Path = MANIFEST_DIR) -> list[str]:
    """Return model-registry problems; an empty list means the registry is valid."""
    problems: list[str] = []
    path = manifest_dir / "model_registry.csv"
    if not path.is_file():
        return [f"missing {path}"]

    registry = pd.read_csv(path)
    missing = REGISTRY_COLUMNS - set(registry.columns)
    if missing:
        return [f"model_registry.csv: missing columns {sorted(missing)}"]
    if registry.empty:
        return ["model_registry.csv: registry is empty"]
    if registry["model_id"].duplicated().any():
        problems.append("model_registry.csv: model_id values are not unique")
    for column in REGISTRY_COLUMNS:
        if _blank(registry[column]).any():
            problems.append(f"model_registry.csv: {column} values must not be blank")

    bad_hash = ~registry["sha256"].astype(str).str.fullmatch(
        re.compile(r"[0-9a-fA-F]{64}")
    )
    if bad_hash.any():
        problems.append("model_registry.csv: sha256 values must be 64 hexadecimal characters")
    if (pd.to_numeric(registry["bytes"], errors="coerce").fillna(0) <= 0).any():
        problems.append("model_registry.csv: bytes values must be positive integers")

    checkpoint_files = registry["checkpoint_file"].astype(str)
    if checkpoint_files.duplicated().any():
        problems.append("model_registry.csv: checkpoint_file values are not unique")
    for column in ["checkpoint_file", "legacy_checkpoint_file"]:
        if registry[column].astype(str).map(lambda value: Path(value).name != value).any():
            problems.append(f"model_registry.csv: {column} values must be filenames")

    missing_models = PRIMARY_MODEL_IDS - set(registry["model_id"])
    if missing_models:
        problems.append(
            "model_registry.csv: missing primary model IDs "
            + ", ".join(sorted(missing_models))
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-populated",
        action="store_true",
        help="retained for command compatibility; the registry is always required",
    )
    args = parser.parse_args()
    problems = validate(args.require_populated)
    if problems:
        raise SystemExit("Model registry validation failed:\n- " + "\n- ".join(problems))
    print("Model registry validated.")


if __name__ == "__main__":
    main()
