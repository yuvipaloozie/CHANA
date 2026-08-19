# Reproducibility status

## Current status

| Component | Status | Evidence/location |
|---|---|---|
| Historical inference and cross-evaluation notebooks | Present | `notebooks/` |
| Historical domain generation/training exports | Present | `notebooks/legacy/python_exports/` |
| Deterministic preprocessing and postprocessing | Implemented and unit tested | `src/chana/` |
| Architecture builders | Extracted from supplied exports | `src/chana/models/` |
| One-image inference CLI | Implemented; requires verified weights | `scripts/predict.py` |
| Dataset/split/domain manifests | Schemas only | `manifests/` |
| Checkpoint registry and hashes | Schema only | `manifests/model_registry.csv` |
| Public source data behind figures/tables | Not yet deposited | `paper/source_data/` |
| Redistributable sample and expected output | Not yet deposited | `sample_data/` |
| Exact end-to-end retraining CLI | Not yet refactored | legacy exports plus `docs/TRAINING.md` |
| Permanent DOI archive and final license | Not yet assigned | release blockers |

## Levels of reproduction

1. **Smoke test:** run preprocessing, mask postprocessing, and metrics on synthetic arrays via `pytest`.
2. **Inference reproduction:** obtain a verified checkpoint, install the matching TensorFlow environment, and run `scripts/predict.py` on the public sample.
3. **Result reproduction:** use the frozen test manifest and all six verified checkpoints to regenerate source-data tables and compare checksums.
4. **Training reproduction:** recreate each domain, retrain baseline and curriculum models with logged seeds/versions, and evaluate only once on the locked test set.

The repository currently supports level 1 and provides the code skeleton for level 2. Levels 2–4 require the pending artifact deposits and manifest completion.

## Minimum release additions

- exact file-level split and scan identifiers;
- six final model checkpoints plus curriculum checkpoint provenance;
- SHA-256 digests and archive URLs;
- actual source-data CSVs for every graph/table;
- training logs showing selected epochs and random seeds;
- one public example with expected output;
- final license, manuscript citation, and archived release DOI.
