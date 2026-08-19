"""Resolve CHANA checkpoints by semantic model ID and verify their bytes."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointSpec:
    model_id: str
    architecture: str
    training_regime: str
    checkpoint_file: str
    legacy_checkpoint_file: str
    bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_registry(path: Path) -> dict[str, CheckpointSpec]:
    """Load a model registry keyed by stable semantic model ID."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    registry: dict[str, CheckpointSpec] = {}
    for row in rows:
        model_id = row["model_id"].strip()
        if model_id in registry:
            raise ValueError(f"duplicate model_id in checkpoint registry: {model_id}")
        registry[model_id] = CheckpointSpec(
            model_id=model_id,
            architecture=row["architecture"].strip(),
            training_regime=row["training_regime"].strip(),
            checkpoint_file=row["checkpoint_file"].strip(),
            legacy_checkpoint_file=row.get("legacy_checkpoint_file", "").strip(),
            bytes=int(row["bytes"]),
            sha256=row["sha256"].strip().lower(),
        )
    return registry


def get_checkpoint_spec(path: Path, model_id: str) -> CheckpointSpec:
    registry = load_checkpoint_registry(path)
    try:
        return registry[model_id]
    except KeyError as error:
        raise ValueError(
            f"unknown model ID {model_id!r}; choose from {sorted(registry)}"
        ) from error


def _safe_filename(value: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"checkpoint registry value is not a filename: {value!r}")
    return value


def resolve_checkpoint_path(spec: CheckpointSpec, weights_dir: Path) -> Path:
    """Resolve the logical release name first, then its historical alias."""
    weights_dir = Path(weights_dir)
    names = [_safe_filename(spec.checkpoint_file)]
    if spec.legacy_checkpoint_file:
        names.append(_safe_filename(spec.legacy_checkpoint_file))
    for name in dict.fromkeys(names):
        candidate = weights_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing checkpoint for {spec.model_id}; expected one of {names} in {weights_dir}"
    )


def verify_checkpoint_file(spec: CheckpointSpec, path: Path) -> None:
    """Reject a checkpoint unless its byte size and SHA-256 match the registry."""
    path = Path(path)
    observed_size = path.stat().st_size
    if observed_size != spec.bytes:
        raise ValueError(
            f"file-size mismatch for {spec.model_id}: {observed_size} != {spec.bytes}"
        )
    observed_hash = sha256_file(path)
    if observed_hash != spec.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {spec.model_id}: {observed_hash} != {spec.sha256}"
        )


def resolve_and_verify_checkpoint(
    registry_path: Path, model_id: str, weights_dir: Path
) -> tuple[CheckpointSpec, Path]:
    spec = get_checkpoint_spec(registry_path, model_id)
    path = resolve_checkpoint_path(spec, weights_dir)
    verify_checkpoint_file(spec, path)
    return spec, path
