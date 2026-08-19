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
        [["unet-baseline", "unet", "baseline", "unet.h5", "legacy.h5", 1, "a" * 64, "test-env"]],
        columns=[
            "model_id", "architecture", "training_regime", "checkpoint_file",
            "legacy_checkpoint_file", "bytes", "sha256", "verified_environment",
        ],
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


def test_primary_model_registry_contains_six_distinct_checkpoints():
    root = Path(__file__).resolve().parents[1]
    registry = pd.read_csv(root / "manifests" / "model_registry.csv")

    assert _validator_module().PRIMARY_MODEL_IDS <= set(registry["model_id"])
    assert registry["sha256"].is_unique
    assert (registry["bytes"] > 0).all()


def test_pseudolabel_teacher_is_registered_separately_from_evaluation_models():
    root = Path(__file__).resolve().parents[1]
    registry = pd.read_csv(root / "manifests" / "model_registry.csv").set_index(
        "model_id"
    )
    teacher = registry.loc["transunet_pseudolabel_teacher"]

    assert teacher["checkpoint_file"] == "transunet_pseudolabel_teacher.weights.h5"
    assert teacher["legacy_checkpoint_file"] == "transunet_stage3_real_finetune.weights.h5"
    assert teacher["sha256"] == "26adf1fdb7afcfb3600f8203234ca324b850ccb22b93431448e80d86a5a11fb5"


def test_primary_model_ids_are_bound_to_hash_verified_regimes():
    root = Path(__file__).resolve().parents[1]
    registry = pd.read_csv(root / "manifests" / "model_registry.csv").set_index(
        "model_id"
    )
    expected = {
        "unet_baseline": (
            "unet_baseline.weights.h5",
            "Unet_DenseNet_Domain.weights.h5",
            "00e22d5af8ac21353f5d30cf9711d89f5902c7a9968dce9481507dccee2875f2",
        ),
        "unet_curriculum": (
            "unet_curriculum.weights.h5",
            "Unet_DenseNet_no_Domain.weights.h5",
            "020aef4a35d38a6251530acfde8cbdca9a3487a2f7e17ffc7a45391589b26498",
        ),
        "unetpp_baseline": (
            "unetpp_baseline.weights.h5",
            "UNetPlusPlus_Domain.weights.h5",
            "271c2d90dde658853615037aa1fb7158d5b0d8a9a48c30dd0a7989b3418573e4",
        ),
        "unetpp_curriculum": (
            "unetpp_curriculum.weights.h5",
            "Unetplusplus_no_Domain.weights.h5",
            "a9a18e22455bfb20caddb5f714b7290ba3e020f8863702977606d44768e3dcfd",
        ),
        "transunet_baseline": (
            "transunet_baseline.weights.h5",
            "transunet_domain.weights.h5",
            "d0927f39a307f69b541113fbfb23781c06c0b9ee853efe2f11836e67bb196fa1",
        ),
        "transunet_curriculum": (
            "transunet_curriculum.weights.h5",
            "transunet_no_domain.weights.h5",
            "4e6300c5f389db16fcea4b3a586855570a873efecbbe0c7ff39bc2fa73788059",
        ),
    }

    observed = {
        model_id: (
            registry.loc[model_id, "checkpoint_file"],
            registry.loc[model_id, "legacy_checkpoint_file"],
            registry.loc[model_id, "sha256"],
        )
        for model_id in expected
    }
    assert observed == expected
