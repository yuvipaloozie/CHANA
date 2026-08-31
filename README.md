# CHANA

**Cell Histology Automated Neural Network Analyzer**

<p align="left">
  <a href="https://github.com/yuvipaloozie/CHANA/actions/workflows/ci.yml">
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" height="30">
  <img alt="TensorFlow 2.16.2" src="https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" height="30">
  <img alt="Git LFS model weights" src="https://img.shields.io/badge/Model_Weights-Git_LFS-F64935?style=for-the-badge&logo=git&logoColor=white" height="30">
  <img alt="Research software" src="https://img.shields.io/badge/Status-Research_Software-4C72B0?style=for-the-badge" height="30">
</p>

CHANA is a semantic-segmentation and morphometric-analysis workflow for
TRAP-stained bright-field osteoclast cultures. Manual counting and segmentation
of mature osteoclasts is time-consuming and observer-dependent. 
Deep learning techniques to automate costeoclast segmentation tasks, specifically, 
require significant volumes of source data to learn the diverse presentations 
of multinucleated osteoclasts.  CHANA tests whether sequential training on
diffusion-derived, copy-paste, pseudo-labelled, and expert-labelled images can
improve performance when expert hand-annotation of osteoclasts is limited.

The repository compares U-Net, U-Net++, and TransUNet trained either on
expert-labelled images alone (baseline) or with the four-stage curriculum. It
includes preprocessing and synthetic & copy-paste data-generation code, 
six evaluation checkpoints, the pseudo-labelling teacher,  tiled inference through final masks and object
measurements, a cleared example, and the machine-readable results currently
available.

> Research use only. The results support technical feasibility for image
> segmentation and measurement of osteoclasts; CHANA has not been validated for clinical use,
> treatment decisions, or prospective drug screening.

## Workflow

![CHANA workflow: preprocessing, three segmentation architectures, domain curriculum, and held-out evaluation](paper/figures/figure_1_overview.png)


Bright-field images are preprocessed and tiled to 512 × 512 pixels. The six
registered models produce foreground probabilities, which are thresholded at
`> 0.5`; hole filling and watershed then separate touching regions for counts
and morphometry. The final overview panel is shown above. The reported
evaluation uses a fixed 281-image holdout, but the deposited records do not
establish scan-, well-, or biological-replicate independence.

## Condensed results

The values below are transcribed from the final main tables. Full confidence
intervals, definitions, all five main/supplementary tables, and direct CSV links
are in [`paper/README.md`](paper/README.md).

| Architecture | Training | Mean IoU | Mean Dice | Median finite HD95 (px) | Mean object F1 | Count MAE |
|---|---|---:|---:|---:|---:|---:|
| U-Net | Baseline | 0.607 | 0.741 | 71.09 | 0.783 | 2.49 |
| U-Net | Curriculum | 0.634 | 0.765 | 61.40 | 0.819 | 2.23 |
| U-Net++ | Baseline | 0.644 | 0.773 | 55.01 | 0.804 | 2.12 |
| **U-Net++** | **Curriculum** | **0.669** | **0.793** | **42.25** | **0.842** | **1.90** |
| TransUNet | Baseline | 0.603 | 0.740 | 64.10 | 0.769 | 2.32 |
| TransUNet | Curriculum | 0.623 | 0.757 | 61.63 | 0.777 | 2.26 |

All configurations were evaluated on the same 281 held-out image tiles. These
are manuscript summary values, not a substitute for the image-level data needed
to regenerate confidence intervals and plots.

## Selected results and interpretation

<details open>
<summary><strong>Domain curriculum and six-model evaluation panels (Figures 2B–4C)</strong></summary>

<h4>Curriculum-domain feature space</h4>
<table>
  <tr>
    <td width="50%" align="center" bgcolor="#ffffff">
      <img src="paper/figures/figure_2b_domain_umap.png" alt="UMAP projection of the four curriculum domains" width="100%"><br>
      <sub><strong>Figure 2B.</strong> UMAP projection of fixed ResNet-50 feature embeddings from the four curriculum domains.</sub>
    </td>
    <td width="50%" align="center" bgcolor="#ffffff">
      <img src="paper/figures/figure_2c_feature_distance.png" alt="Feature distance from generated domains to expert-real images" width="100%"><br>
      <sub><strong>Figure 2C.</strong> Feature-space distance from non-real domains to expert-real images.</sub>
    </td>
  </tr>
</table>

<h4>Sequential curriculum checkpoints</h4>
<table>
  <tr>
    <td width="50%" align="center"><img src="paper/figures/figure_3a_curriculum_iou.png" alt="U-Net++ IoU across curriculum stages" width="100%"><br><sub><strong>Figure 3A.</strong> Mean IoU.</sub></td>
    <td width="50%" align="center"><img src="paper/figures/figure_3b_curriculum_dice.png" alt="U-Net++ Dice across curriculum stages" width="100%"><br><sub><strong>Figure 3B.</strong> Mean Dice.</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><img src="paper/figures/figure_3c_curriculum_object_f1.png" alt="U-Net++ object F1 across curriculum stages" width="100%"><br><sub><strong>Figure 3C.</strong> Mean object F1.</sub></td>
    <td width="50%" align="center"><img src="paper/figures/figure_3d_curriculum_count_mae.png" alt="U-Net++ count mean absolute error across curriculum stages" width="100%"><br><sub><strong>Figure 3D.</strong> Mean absolute count error; lower is better.</sub></td>
  </tr>
</table>

<h4>Held-out segmentation performance</h4>
<p align="center">
  <img src="paper/figures/figure_4a_pixel_precision_recall.png" alt="Pixel precision-recall curves for all six model configurations" width="560"><br>
  <sub><strong>Figure 4A.</strong> Pixel precision-recall curves for all six model configurations.</sub>
</p>
<table>
  <tr>
    <td width="50%" align="center"><img src="paper/figures/figure_4b_iou_distribution.png" alt="Image-level IoU distributions for all six model configurations" width="100%"><br><sub><strong>Figure 4B.</strong> Image-level IoU distributions.</sub></td>
    <td width="50%" align="center"><img src="paper/figures/figure_4c_hd95_distribution.png" alt="Finite HD95 distributions for all six model configurations" width="100%"><br><sub><strong>Figure 4C.</strong> Finite HD95 distributions; lower is better.</sub></td>
  </tr>
</table>

</details>

<p align="center">
  <img src="paper/figures/supplementary_figure_s4_model_interpretation.png" alt="Curriculum U-Net++ input, expert mask, foreground probability, probability entropy, and gradient activation" width="100%">
</p>
<p align="center"><sub><strong>Exploratory model-output interpretation.</strong> A representative curriculum U-Net++ field is shown with its expert mask, foreground probability, probability entropy, and gradient-based activation.</sub></p>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="paper/figures/figure_6a_object_recall_by_area.png" alt="Object recall by reference-object area" width="100%"><br>
      <sub><strong>Object recall by area.</strong> Recall across small, intermediate, and large expert-mask-derived objects.</sub>
    </td>
    <td width="50%" align="center">
      <img src="paper/figures/figure_6b_iou_by_density.png" alt="Image-level IoU by reference-object density" width="100%"><br>
      <sub><strong>IoU by field density.</strong> Image-level IoU across low, moderate, and dense reference-count strata.</sub>
    </td>
  </tr>
</table>

<p align="center">
  <img src="paper/figures/figure_4d_paired_iou_difference.png" alt="Paired curriculum-minus-baseline mean IoU differences" width="440"><br>
  <sub><strong>Paired curriculum effect.</strong> Curriculum-minus-baseline mean IoU differences with 95% image-bootstrap intervals.</sub>
</p>

These panels are descriptive views of the finalized analyses. Exact reported
values remain in [`paper/README.md`](paper/README.md) and its linked CSVs; plots
should not be used to recover numerical values.

## Quick start

Requirements: Git LFS, Python 3.10, and about 2.5 GB of free space for the
checkpoints.

```bash
git lfs install
git clone https://github.com/yuvipaloozie/CHANA.git
cd CHANA
git lfs pull

conda env create -f environment/inference.yml
conda activate chana-inference
python -m pip install -e .
python scripts/validate_checkpoints.py --weights-dir models --hash-only
```

Run the complete pipeline on the included image:

```bash
python scripts/predict.py \
  --input sample_data/public_example/input_image.tif \
  --output-dir outputs/public_example \
  --model-id unetpp_curriculum \
  --weights-dir models
```

The command writes the foreground probability array, binary mask, watershed
labels, and an object-measurement CSV.

Compare the U-Net++ baseline and curriculum checkpoints on the included
image-mask pair:

```bash
python scripts/compare_models.py \
  --pairs-csv sample_data/public_example/pairs.csv \
  --output-dir outputs/unetpp_comparison \
  --weights-dir models \
  --save-predictions
```

This verifies both checkpoint hashes and reports per-image and summary IoU,
Dice, pixel average precision, HD95, object precision/recall/F1, and count error.
Use `--input-stage preprocessed` for images that have already passed through V9.

## Registered models

| Model ID | Architecture | Training | Checkpoint |
|---|---|---|---|
| `unet_baseline` | U-Net / DenseNet121 | Expert-real only | `unet_baseline.weights.h5` |
| `unet_curriculum` | U-Net / DenseNet121 | Sequential curriculum | `unet_curriculum.weights.h5` |
| `unetpp_baseline` | U-Net++ / ResNet50 | Expert-real only | `unetpp_baseline.weights.h5` |
| `unetpp_curriculum` | U-Net++ / ResNet50 | Sequential curriculum | `unetpp_curriculum.weights.h5` |
| `transunet_baseline` | TransUNet / EfficientNetB0 | Expert-real only | `transunet_baseline.weights.h5` |
| `transunet_curriculum` | TransUNet / EfficientNetB0 | Sequential curriculum | `transunet_curriculum.weights.h5` |

[`manifests/model_registry.csv`](manifests/model_registry.csv) is the
 mapping from model ID to checkpoint filename,
legacy filename, size, and SHA-256. The separately registered
`transunet_pseudolabel_teacher.weights.h5` generated pseudo-labels and is not one
of the six final evaluation models.

## Training and data generation

The curriculum transfers weights through 3,000 diffusion-derived pairs, 1,500
copy-paste pairs, 2,058 pseudo-labelled real pairs, and final expert-real
training. Baselines use expert-real training only. Full settings and phase order
are in [`docs/TRAINING.md`](docs/TRAINING.md).

The original Colab exports remain under
[`notebooks/legacy/python_exports/`](notebooks/legacy/python_exports/). Set their
Drive input/output paths before running them.

| Task | Script |
|---|---|
| RGB preprocessing | [`chana_preprocessing_rgb.py`](notebooks/legacy/python_exports/chana_preprocessing_rgb.py) |
| Diffusion-domain refinement | [`chana_diffusionv2.py`](notebooks/legacy/python_exports/chana_diffusionv2.py) |
| Copy-paste generation | [`chana_copy_paste_gen.py`](notebooks/legacy/python_exports/chana_copy_paste_gen.py) |
| Pseudo-labelling | [`chana_pseudo_labelling.py`](notebooks/legacy/python_exports/chana_pseudo_labelling.py) |
| U-Net baseline / curriculum | [`without domains`](notebooks/legacy/python_exports/chana_unet_without_domains.py) / [`with domains`](notebooks/legacy/python_exports/chana_unet_with_domains.py) |
| U-Net++ baseline / curriculum | [`without domains`](notebooks/legacy/python_exports/chana_unetpp_without_domains.py) / [`with domains`](notebooks/legacy/python_exports/chana_unetpp_with_domains.py) |
| TransUNet baseline / curriculum | [`without domains`](notebooks/legacy/python_exports/chana_transunet_without_domains.py) / [`with domains`](notebooks/legacy/python_exports/chana_transunet_with_domains.py) |

Reusable preprocessing, model builders, inference, postprocessing, and metrics
are under [`src/chana/`](src/chana/). The notebooks under [`notebooks/`](notebooks/)
provide interactive entry points without embedded manuscript figures.

## Repository guide

| Path | Contents |
|---|---|
| `src/chana/` | Reusable preprocessing, models, inference, metrics, and postprocessing |
| `scripts/` | Command-line inference, comparison, and validation |
| `models/` | Six evaluation checkpoints plus the pseudo-label teacher (Git LFS) |
| `environment/` | Pinned inference, training, and generation environments |
| `notebooks/` | Interactive notebooks and original Colab exports |
| `sample_data/` | Cleared example input, mask, and reference outputs |
| `manifests/model_registry.csv` | Canonical checkpoint identity and hashes |
| `paper/` | Figure 1, readable final tables, table CSVs, and available plot data |
| `tests/` | Deterministic unit and integrity tests |

## Validate the repository

```bash
python -m pip install -e ".[test]"
python scripts/validate_manifests.py
python scripts/validate_source_data.py
python scripts/validate_sample_data.py --require-cleared
python -m pytest
```

## Reproducibility scope

The repository currently supports environment creation, hash-checked model
selection, inference through object measurements, baseline/curriculum comparison,
training-script inspection, and validation of the deposited example and result
tables.

Before archival release, add the exact image-to-split mapping with stable image
and source-scan identifiers, plus the remaining machine-readable image/object
data behind the final plots and confidence intervals. Until that mapping is
available, describe the evaluation set only as a fixed 281-image holdout. Final
licensing, release tagging, and DOI archiving are separate release tasks.

## Citation

Citation metadata are in [`CITATION.cff`](CITATION.cff). The manuscript itself
is intentionally not linked from this repository at this stage.
