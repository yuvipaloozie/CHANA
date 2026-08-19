# Contributing

CHANA is currently a prepublication research repository. Please open an issue before making substantial changes to model definitions, preprocessing, postprocessing, or reported analyses.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/validate_manifests.py
python -m pytest
```

## Contribution requirements

- Keep the locked test set outside model/threshold selection.
- Preserve stable image and model identifiers in outputs.
- Add or update tests for reusable code changes.
- Record new dependencies in `pyproject.toml` and the relevant environment file.
- Do not commit restricted data, local Drive paths, credentials, or personally identifying information. Large approved checkpoints must use Git LFS.
- For numerical manuscript changes, include the machine-readable source data and explain whether the analysis is paired and what the independent unit is.
- Do not relabel historical checkpoints without verifying their hashes and expected outputs.

The final software license is pending institutional approval; external reuse terms are therefore not granted until a `LICENSE` file is added.
