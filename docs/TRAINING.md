# Training protocol

This document records the protocol represented in the supplied notebook exports. The original exports remain under `notebooks/legacy/python_exports/` and require their Colab/Drive paths to be set before execution.

## Shared settings

- Input: 512 × 512 RGB images after V9 preprocessing.
- Batch size: 16.
- Optimizer: AdamW with weight decay 1 × 10⁻⁴.
- Learning-rate schedule: cosine decay.
- Trusted-label loss: focal Tversky (`alpha` FN = 0.3, `beta` FP = 0.7, `gamma` = 1.5).
- Noisy-label loss: binary cross-entropy with label smoothing 0.1 where invoked by the export.
- Augmentation: horizontal and vertical flips (p = 0.5 each), rotation up to ±20° (p = 0.5), brightness/contrast (p = 0.2), and coarse dropout (up to six 24 × 24 holes; p = 0.3).

## Baseline training

Each architecture is trained on the same expert-real training partition without the preceding curriculum domains.

| Architecture | Maximum epochs | Initial learning rate |
|---|---:|---:|
| U-Net | 400 | 1 × 10⁻⁴ |
| U-Net++ | 500 | 1 × 10⁻⁴ |
| TransUNet | 400 | 1 × 10⁻⁵ |

The selected epoch for each released model should be added from the historical logs when available.

## Sequential curriculum

Within each architecture, weights are transferred in this order:

| Phase | Domain | Training images | Epochs | Initial learning rate |
|---|---|---:|---:|---:|
| 1 | Diffusion-derived | 3,000 | 40 | 1 × 10⁻⁴ |
| 2 | Copy-paste | 1,500 | 80 | 5 × 10⁻⁵ |
| 2.5 | Pseudo-labeled real | 2,058 | 50 | 2 × 10⁻⁵ |
| 4 | Expert-labeled real | 1,629 | 200 | 1 × 10⁻⁵ |

The supplemental Word table reports 1,910 expert-labelled images used because
that total includes 1,629 training and 281 validation images; optimization uses
the 1,629-image training partition.

Pseudo-labels were produced by the fixed expert-real-fine-tuned TransUNet
checkpoint `transunet_pseudolabel_teacher.weights.h5` (historical filename
`transunet_stage3_real_finetune.weights.h5`). They were not produced by each
architecture's immediately preceding checkpoint.

## Evaluation boundary

Validation data may be used for checkpoint selection and threshold specification. The 281-image test set must not be used for model selection, threshold tuning, or figure-driven iteration. Final analyses should load the frozen split manifest and verified model registry.

## Reproduction commands

The training files are original Colab exports rather than parameterized command-line programs. Set the paths in each export, use the matching environment, and record the resulting checkpoint SHA-256 and selected epoch.

The essential exports are:

- `chana_unet_without_domains.py` and `chana_unet_with_domains.py`;
- `chana_unetpp_without_domains.py` and `chana_unetpp_with_domains.py`;
- `chana_transunet_without_domains.py` and `chana_transunet_with_domains.py`;
- `chana_diffusionv2.py`, `chana_copy_paste_gen.py`, and
  `chana_pseudo_labelling.py` for the curriculum data phases.
- `chana_preprocessing_rgb.py` for the original V9 enhancement workflow.

All are under `notebooks/legacy/python_exports/`. They retain Colab `!pip`
cells; run them in Colab or remove those notebook-only lines for local use.
