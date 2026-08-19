#!/usr/bin/env python
"""Validate CHANA manifest schemas and cross-file identifiers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"

SCHEMAS = {
    "dataset_manifest.csv": {
        "image_id", "image_path", "mask_path", "domain", "split", "scan_id", "source_id"
    },
    "split_manifest.csv": {"image_id", "split", "scan_id", "random_seed", "notes"},
    "domain_manifest.csv": {"image_id", "domain", "source_id", "label_source", "notes"},
    "model_registry.csv": {
        "model_id", "architecture", "training_regime", "checkpoint_file", "sha256", "notes"
    },
}

EXPECTED_COUNTS = {
    "diffusion": 3000,
    "copy_paste": 1500,
    "pseudo_label": 2058,
    "expert_train": 1629,
    "expert_validation": 281,
    "expert_test": 281,
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
    problems: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for filename, required in SCHEMAS.items():
        path = manifest_dir / filename
        if not path.is_file():
            problems.append(f"missing {path.relative_to(ROOT)}")
            continue
        frame = pd.read_csv(path)
        frames[filename] = frame
        missing = required - set(frame.columns)
        if missing:
            problems.append(f"{filename}: missing columns {sorted(missing)}")
        if require_populated and frame.empty:
            problems.append(f"{filename}: must be populated for a release")

    dataset = frames.get("dataset_manifest.csv")
    if dataset is not None and not dataset.empty and not (SCHEMAS["dataset_manifest.csv"] - set(dataset.columns)):
        if dataset["image_id"].duplicated().any():
            problems.append("dataset_manifest.csv: image_id values are not unique")
        if _blank(dataset["image_id"]).any():
            problems.append("dataset_manifest.csv: image_id values must not be blank")
        counts = {
            "diffusion": int((dataset.domain == "diffusion").sum()),
            "copy_paste": int((dataset.domain == "copy_paste").sum()),
            "pseudo_label": int((dataset.domain == "pseudo_label").sum()),
            "expert_train": int(((dataset.domain == "expert_real") & (dataset.split == "train")).sum()),
            "expert_validation": int(((dataset.domain == "expert_real") & (dataset.split == "validation")).sum()),
            "expert_test": int(((dataset.domain == "expert_real") & (dataset.split == "test")).sum()),
        }
        for group, expected in EXPECTED_COUNTS.items():
            if counts[group] != expected:
                problems.append(f"dataset_manifest.csv: {group} count {counts[group]} != {expected}")

        expert = dataset[dataset["domain"] == "expert_real"]
        if require_populated and _blank(expert["scan_id"]).any():
            problems.append("dataset_manifest.csv: expert_real rows require scan_id values for release")

    split = frames.get("split_manifest.csv")
    if split is not None and not split.empty and not (SCHEMAS["split_manifest.csv"] - set(split.columns)):
        if split["image_id"].duplicated().any():
            problems.append("split_manifest.csv: image_id values are not unique")
        if _blank(split["image_id"]).any():
            problems.append("split_manifest.csv: image_id values must not be blank")
        if require_populated and _blank(split["scan_id"]).any():
            problems.append("split_manifest.csv: scan_id values are required for release")

        nonblank_scans = split.loc[~_blank(split["scan_id"])].copy()
        development_scans = set(nonblank_scans.loc[nonblank_scans["split"].isin(["train", "validation"]), "scan_id"])
        test_scans = set(nonblank_scans.loc[nonblank_scans["split"] == "test", "scan_id"])
        overlap = sorted(development_scans & test_scans)
        if overlap:
            problems.append(
                "split_manifest.csv: scan_id overlap between development and test: "
                + ", ".join(overlap[:10])
            )

    domain = frames.get("domain_manifest.csv")
    if domain is not None and not domain.empty and not (SCHEMAS["domain_manifest.csv"] - set(domain.columns)):
        if domain["image_id"].duplicated().any():
            problems.append("domain_manifest.csv: image_id values are not unique")
        if _blank(domain["image_id"]).any():
            problems.append("domain_manifest.csv: image_id values must not be blank")

    registry = frames.get("model_registry.csv")
    if registry is not None and not registry.empty and not (SCHEMAS["model_registry.csv"] - set(registry.columns)):
        if registry["model_id"].duplicated().any():
            problems.append("model_registry.csv: model_id values are not unique")
        for column in ["model_id", "architecture", "training_regime", "checkpoint_file", "sha256"]:
            if _blank(registry[column]).any():
                problems.append(f"model_registry.csv: {column} values must not be blank")
        bad_hash = ~registry["sha256"].fillna("").astype(str).str.fullmatch(re.compile(r"[0-9a-fA-F]{64}"))
        if bad_hash.any():
            problems.append("model_registry.csv: sha256 values must be 64 hexadecimal characters")
        if "bytes" in registry and (
            pd.to_numeric(registry["bytes"], errors="coerce").fillna(0) <= 0
        ).any():
            problems.append("model_registry.csv: bytes values must be positive integers")
        checkpoint_files = registry["checkpoint_file"].fillna("").astype(str)
        if checkpoint_files.duplicated().any():
            problems.append("model_registry.csv: checkpoint_file values are not unique")
        if checkpoint_files.map(lambda value: Path(value).name != value).any():
            problems.append("model_registry.csv: checkpoint_file values must be filenames")
        if "legacy_checkpoint_file" in registry:
            legacy = registry["legacy_checkpoint_file"].fillna("").astype(str)
            if legacy.str.strip().eq("").any():
                problems.append(
                    "model_registry.csv: legacy_checkpoint_file values must not be blank"
                )
            if legacy.map(lambda value: Path(value).name != value).any():
                problems.append(
                    "model_registry.csv: legacy_checkpoint_file values must be filenames"
                )
        if require_populated:
            missing_models = PRIMARY_MODEL_IDS - set(registry["model_id"])
            if missing_models:
                problems.append(
                    "model_registry.csv: missing primary model IDs "
                    + ", ".join(sorted(missing_models))
                )
            if "archive_url" not in registry or _blank(registry["archive_url"]).any():
                problems.append("model_registry.csv: archive_url values are required for release")
            if "load_status" not in registry or (
                registry["load_status"].astype(str) != "architecture_load_verified"
            ).any():
                problems.append(
                    "model_registry.csv: every checkpoint must be architecture-load verified"
                )

    if dataset is not None and not dataset.empty and split is not None and not split.empty:
        expert_ids = set(dataset.loc[dataset["domain"] == "expert_real", "image_id"])
        split_ids = set(split["image_id"])
        if expert_ids != split_ids:
            problems.append("split_manifest.csv: image_id set must equal expert_real IDs in dataset_manifest.csv")
        joined = dataset[["image_id", "split", "scan_id"]].merge(
            split[["image_id", "split", "scan_id"]], on="image_id", suffixes=("_dataset", "_split")
        )
        if (joined["split_dataset"].astype(str) != joined["split_split"].astype(str)).any():
            problems.append("split_manifest.csv: split assignments disagree with dataset_manifest.csv")
        comparable_scan = ~_blank(joined["scan_id_dataset"]) & ~_blank(joined["scan_id_split"])
        if (
            joined.loc[comparable_scan, "scan_id_dataset"].astype(str)
            != joined.loc[comparable_scan, "scan_id_split"].astype(str)
        ).any():
            problems.append("split_manifest.csv: scan_id values disagree with dataset_manifest.csv")

    if dataset is not None and not dataset.empty and domain is not None and not domain.empty:
        dataset_ids = set(dataset["image_id"])
        domain_ids = set(domain["image_id"])
        if dataset_ids != domain_ids:
            problems.append("domain_manifest.csv: image_id set must equal dataset_manifest.csv")
        joined = dataset[["image_id", "domain"]].merge(
            domain[["image_id", "domain"]], on="image_id", suffixes=("_dataset", "_domain")
        )
        if (joined["domain_dataset"].astype(str) != joined["domain_domain"].astype(str)).any():
            problems.append("domain_manifest.csv: domain assignments disagree with dataset_manifest.csv")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-populated", action="store_true")
    args = parser.parse_args()
    problems = validate(args.require_populated)
    if problems:
        raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(problems))
    print("Manifest schemas validated.")


if __name__ == "__main__":
    main()
