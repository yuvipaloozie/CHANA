import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_active_evaluation_notebook_uses_canonical_checkpoint_names():
    source = _source(ROOT / "notebooks" / "Model_Cross_Evaluation.ipynb")
    expected = {
        "unet_baseline": "unet_baseline.weights.h5",
        "unet_curriculum": "unet_curriculum.weights.h5",
        "unetpp_baseline": "unetpp_baseline.weights.h5",
        "unetpp_curriculum": "unetpp_curriculum.weights.h5",
        "transunet_baseline": "transunet_baseline.weights.h5",
        "transunet_curriculum": "transunet_curriculum.weights.h5",
    }
    for model_id, checkpoint_file in expected.items():
        assert f'"model_id": "{model_id}"' in source
        assert f'"file": "{checkpoint_file}"' in source


def test_active_inference_notebook_selects_canonical_curriculum_checkpoint():
    source = _source(ROOT / "notebooks" / "CHANA_Inference_Notebook.ipynb")

    assert "MODEL_ID = 'unetpp_curriculum'" in source
    assert "model_weights/unetpp_curriculum.weights.h5" in source
    assert "Unetplusplus_no_Domain.weights.h5" not in source
