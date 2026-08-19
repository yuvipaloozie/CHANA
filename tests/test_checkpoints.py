import hashlib
from pathlib import Path

import pytest

from chana.checkpoints import (
    get_checkpoint_spec,
    resolve_and_verify_checkpoint,
    resolve_checkpoint_path,
)


def _write_registry(path: Path, digest: str, size: int) -> None:
    path.write_text(
        "model_id,architecture,training_regime,checkpoint_file,"
        "legacy_checkpoint_file,bytes,sha256\n"
        f"unetpp_curriculum,unetpp,sequential_domain_curriculum,"
        f"unetpp_curriculum.weights.h5,misleading_no_Domain.weights.h5,"
        f"{size},{digest}\n",
        encoding="utf-8",
    )


def test_semantic_checkpoint_resolution_prefers_canonical_name(tmp_path):
    content = b"hash-verified checkpoint fixture"
    digest = hashlib.sha256(content).hexdigest()
    registry = tmp_path / "registry.csv"
    _write_registry(registry, digest, len(content))
    (tmp_path / "misleading_no_Domain.weights.h5").write_bytes(content)
    (tmp_path / "unetpp_curriculum.weights.h5").write_bytes(content)

    spec, resolved = resolve_and_verify_checkpoint(
        registry, "unetpp_curriculum", tmp_path
    )

    assert spec.architecture == "unetpp"
    assert resolved.name == "unetpp_curriculum.weights.h5"


def test_semantic_checkpoint_resolution_supports_preserved_legacy_name(tmp_path):
    content = b"legacy-named checkpoint fixture"
    digest = hashlib.sha256(content).hexdigest()
    registry = tmp_path / "registry.csv"
    _write_registry(registry, digest, len(content))
    legacy = tmp_path / "misleading_no_Domain.weights.h5"
    legacy.write_bytes(content)

    spec = get_checkpoint_spec(registry, "unetpp_curriculum")
    assert resolve_checkpoint_path(spec, tmp_path) == legacy


def test_semantic_checkpoint_resolution_rejects_wrong_bytes(tmp_path):
    expected = b"expected"
    registry = tmp_path / "registry.csv"
    _write_registry(registry, hashlib.sha256(expected).hexdigest(), len(expected))
    (tmp_path / "unetpp_curriculum.weights.h5").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_and_verify_checkpoint(registry, "unetpp_curriculum", tmp_path)
