import subprocess
import sys
import importlib.util
from pathlib import Path

import pandas as pd


def test_manifest_schemas_validate():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "validate_manifests.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _validator_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("validate_manifests", root / "scripts" / "validate_manifests.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifests(directory: Path, scan_ids=("scan-a", "scan-b")):
    pd.DataFrame(
        [
            ["train-image", "train.tif", "train-mask.tif", "expert_real", "train", scan_ids[0], ""],
            ["test-image", "test.tif", "test-mask.tif", "expert_real", "test", scan_ids[1], ""],
        ],
        columns=["image_id", "image_path", "mask_path", "domain", "split", "scan_id", "source_id"],
    ).to_csv(directory / "dataset_manifest.csv", index=False)
    pd.DataFrame(
        [
            ["train-image", "train", scan_ids[0], 42, ""],
            ["test-image", "test", scan_ids[1], 42, ""],
        ],
        columns=["image_id", "split", "scan_id", "random_seed", "notes"],
    ).to_csv(directory / "split_manifest.csv", index=False)
    pd.DataFrame(
        [
            ["train-image", "expert_real", "", "expert", ""],
            ["test-image", "expert_real", "", "expert", ""],
        ],
        columns=["image_id", "domain", "source_id", "label_source", "notes"],
    ).to_csv(directory / "domain_manifest.csv", index=False)
    pd.DataFrame(
        [["unet-baseline", "unet", "baseline", "unet.h5", "a" * 64, ""]],
        columns=["model_id", "architecture", "training_regime", "checkpoint_file", "sha256", "notes"],
    ).to_csv(directory / "model_registry.csv", index=False)


def test_scan_overlap_is_reported(tmp_path):
    _write_manifests(tmp_path, scan_ids=("shared-scan", "shared-scan"))
    problems = _validator_module().validate(manifest_dir=tmp_path)
    assert any("scan_id overlap" in problem for problem in problems)


def test_bad_checkpoint_hash_is_reported(tmp_path):
    _write_manifests(tmp_path)
    registry = pd.read_csv(tmp_path / "model_registry.csv")
    registry.loc[0, "sha256"] = "filename-is-not-provenance"
    registry.to_csv(tmp_path / "model_registry.csv", index=False)
    problems = _validator_module().validate(manifest_dir=tmp_path)
    assert any("64 hexadecimal" in problem for problem in problems)
