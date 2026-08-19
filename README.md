# CHANA

**Sequential domain-curriculum learning for segmentation and morphometric quantification of cultured osteoclasts**

CHANA is a research pipeline for semantic segmentation of TRAP-stained cultured osteoclasts in bright-field microscopy. It compares U-Net, U-Net++, and TransUNet under conventional expert-real training and a sequential curriculum containing diffusion-derived, copy-paste, pseudo-labeled, and expert-labeled real images. Semantic masks are converted to separated objects for cell counts and morphometric measurements.

> **Prepublication repository.** This branch is being prepared to accompany a manuscript. Model weights, de-identified data/source-data deposits, permanent archive DOI, and the final software license must be added before publication. The current code is for research use and is not a clinical or drug-testing system.

## Study at a glance

| Component | Specification |
|---|---|
| Expert-labeled real images | 2,191 image-mask pairs: 1,629 training, 281 validation, 281 test |
| Curriculum domains | 3,000 diffusion-derived; 1,500 copy-paste; 2,058 pseudo-labeled; 1,629 expert-real training images |
| Architectures | U-Net/DenseNet121, U-Net++/ResNet50, TransUNet/EfficientNetB0 |
| Input | RGB microscopy image, processed as 512 × 512 tiles |
| Output | Foreground probability map, binary semantic mask, separated object labels, count, and morphometrics |
| Primary evaluation | Pixel overlap/discrimination, HD95 boundary error, centroid-matched object metrics, and count agreement |
| Test design | A fixed 281-image holdout set; the final split manifest must document the random split and scan-level independence |

Performance summaries are intentionally omitted here until the final main-table values are deposited as verified machine-readable source data tied to the checkpoint registry and locked split manifest.

## Repository layout

```text
CHANA/
├── configs/                  # Dataset, model, and experiment specifications
├── docs/                     # Data, training, model-card, and release documentation
├── environment/              # Inference/training/generation environments
├── manifests/                # File-level data, split, domain, and checkpoint registries
├── models/                   # Weight-placement instructions (weights are not committed)
├── notebooks/                # Original Colab notebooks and legacy Python exports
├── paper/
│   ├── expected_outputs/     # Expected manuscript-output inventory
│   └── source_data/          # Machine-readable values underlying figures and tables
├── sample_data/              # Instructions for a redistributable smoke-test example
├── scripts/                  # Command-line inference and validation utilities
├── src/chana/                # Reusable preprocessing, models, inference, and metrics
└── tests/                    # Lightweight deterministic checks
```

## Installation

The historical experiments used multiple frameworks. For routine inference and evaluation, use Python 3.10 and the pinned TensorFlow extra:

```bash
git clone https://github.com/yuvipaloozie/CHANA.git
cd CHANA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tensorflow]"
```

Environment specifications are also provided under [`environment/`](environment/). Because the original experiments ran in Google Colab, exact reproduction should use the framework-specific environment matching the relevant script.

## Inference

Place a compatible checkpoint under `models/` and run:

```bash
python scripts/predict.py \
  --input path/to/image.tif \
  --output-dir outputs/example \
  --architecture unetpp \
  --checkpoint models/UNetPlusPlus_checkpoint.weights.h5
```

The command writes the foreground probability array, binary mask, watershed object-label image, and object measurements. Checkpoint identities must be verified against `manifests/model_registry.csv`; filenames alone are not sufficient provenance.

## Reproducing the study

1. Recreate the appropriate environment from `environment/`.
2. Populate and validate the manifests:

   ```bash
   python scripts/validate_manifests.py --require-populated
   ```

3. Obtain data and weights from the manuscript's Data Availability and Code Availability statements.
4. Follow [`docs/TRAINING.md`](docs/TRAINING.md) for baseline and curriculum schedules.
5. Use the locked test manifest for final evaluation. Do not tune thresholds or model choices on the test set.
6. Compare regenerated values against `paper/expected_outputs/` and the source-data deposit.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for what is currently runnable and what remains to be deposited.

## Scientific scope and limitations

- The model segments regions that experts annotated as osteoclasts after visually applying the study definition; it does **not** independently count nuclei.
- Watershed-derived objects are analytical instances inferred from semantic masks, not direct instance annotations.
- Images are two-dimensional bright-field fields from the study acquisition setting. Generalization to other laboratories, microscopes, stains, cell lines, species, or clinical material has not been established.
- The current repository contains no evidence of prospective drug-screening performance.
- A random split can still permit unmodeled acquisition correlation. The final release must include stable file identifiers and scan grouping so readers can verify split independence.

## Citation and archival release

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Before manuscript publication, create an immutable archive (for example, through Zenodo), update the DOI and manuscript citation, deposit the data underlying every graph and table in reusable formats, and add the institutionally approved software license.

## Contributors

Sarah Szabo, Yuvraj Tripathy, Leonard Clark, Kyeong Heo, Vicky Mody, and Shashidharamurthy Taval.
