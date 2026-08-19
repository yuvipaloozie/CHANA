from pathlib import Path

import pytest
import yaml

from chana.sample_validation import SampleValidationError, validate_sample_manifest


MANIFEST = Path("sample_data/public_example/manifest.yml")


def test_public_example_files_match_manifest():
    report = validate_sample_manifest(MANIFEST)

    assert report["shape"] == (512, 512)
    assert report["foreground_pixels"] == 15395
    assert report["foreground_fraction"] == pytest.approx(0.058727264404296875)
    assert report["predicted_foreground_pixels"] == 15087
    assert report["predicted_foreground_fraction"] == pytest.approx(
        0.057552337646484375
    )
    assert report["watershed_label_count"] == 13
    assert report["overlay_checkpoint_model_id"] == "unresolved"


def test_release_clearance_is_explicit():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["redistribution"]["status"] == "cleared"
    validate_sample_manifest(MANIFEST, require_cleared=True)


def test_release_checkpoint_ambiguity_is_explicit():
    with pytest.raises(SampleValidationError, match="checkpoint is unresolved"):
        validate_sample_manifest(MANIFEST, require_checkpoint_linked=True)
