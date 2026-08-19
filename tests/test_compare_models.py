import importlib.util
from pathlib import Path


def _comparison_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "compare_models", root / "scripts" / "compare_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_comparison_pair_resolves_to_cleared_files():
    root = Path(__file__).resolve().parents[1]
    rows = _comparison_module().load_pairs(
        root / "sample_data" / "public_example" / "pairs.csv"
    )

    assert len(rows) == 1
    assert rows[0]["image_id"] == "chana_public_example_001"
    assert Path(rows[0]["image_path"]).name == "input_image.tif"
    assert Path(rows[0]["mask_path"]).name == "reference_mask.tif"
