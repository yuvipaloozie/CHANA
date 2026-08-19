# Reproducibility

## What is included

- pinned inference, training, and generation environments in `environment/`;
- the six baseline/curriculum training exports, plus diffusion, copy-paste, and pseudo-label generation exports, in `notebooks/legacy/python_exports/`;
- reusable preprocessing, model construction, postprocessing, inference, and metrics under `src/chana/`;
- hash-registered metadata for all six final checkpoints in `manifests/model_registry.csv`;
- an end-to-end tiled inference command in `scripts/predict.py`;
- a cleared six-TIFF example in `sample_data/public_example/`;
- final Word-authoritative panel/table inventories in `paper/final_assets/`; and
- deposited source data currently available under `paper/source_data/`.

The six primary weights are stored with Git LFS under logical filenames. The
registry verifies their semantic model ID, byte size, and SHA-256 before use.
Restricted test data remains outside Git.

## Routine inference

```bash
conda env create -f environment/inference.yml
conda activate chana-inference
python -m pip install -e .
python scripts/predict.py --input image.tif --output-dir outputs/example \
  --model-id unetpp_curriculum --weights-dir models
```

`predict.py` performs V9 preprocessing, white padding, 512 x 512 tiling, model
output selection, stitching, strict `> 0.5` thresholding, hole filling,
watershed separation, and object measurement.

## Remaining inputs

- populate stable dataset/split IDs, including `scan_id`, before claiming scan-level independence;
- add the original machine-readable CSV/NPZ data behind the remaining final Word figures and tables;
- add phase checkpoints, the pseudo-label teacher, and training logs if exact retraining is required; and
- link the public six-TIFF example to its exact originating checkpoint.

The final Word documents control reported values and panel selection. The
repository does not claim well-level splitting, drug-testing validation, or
full training reproducibility until the corresponding evidence is available.
