# Training protocol

This document records the protocol represented in the supplied notebook exports. The legacy exports remain under `notebooks/legacy/python_exports/` for auditability; they require path cleanup before general execution.

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

The final release must state the exact early-stopping/checkpoint monitor and selected epoch for each model from the historical logs.

## Sequential curriculum

Within each architecture, weights are transferred in this order:

| Phase | Domain | Images | Epochs | Initial learning rate |
|---|---|---:|---:|---:|
| 1 | Diffusion-derived | 3,000 | 40 | 1 × 10⁻⁴ |
| 2 | Copy-paste | 1,500 | 80 | 5 × 10⁻⁵ |
| 2.5 | Pseudo-labeled real | 2,058 | 50 | 2 × 10⁻⁵ |
| 4 | Expert-labeled real | 1,629 | 200 | 1 × 10⁻⁵ |

Pseudo-labels were produced by a fixed TransUNet teacher fine-tuned on expert-real data. They were not produced by each architecture's immediately preceding checkpoint. This distinction must remain explicit in the paper and model card.

## Evaluation boundary

Validation data may be used for checkpoint selection and threshold specification. The 281-image test set must not be used for model selection, threshold tuning, or figure-driven iteration. Final analyses should load the frozen split manifest and verified model registry.

## Reproduction commands

The current training notebooks are historical Colab exports rather than fully parameterized command-line programs. Exact one-command retraining remains a release blocker. Until refactoring is completed, run the appropriate environment and legacy export with paths replaced by manifest-driven loaders, then document the checkpoint SHA-256 and selected epoch.
