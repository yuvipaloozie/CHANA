# Manuscript results

This page is the compact repository representation of the finalized Word
tables and selected figure panels. The Word documents control the reported
values and panel selection; the linked CSVs provide the table values in
machine-readable form. The manuscript draft itself is not deposited here.

## Available figures and data

| Manuscript item | Repository file | Scope |
|---|---|---|
| Figure 1 | [`figures/figure_1_overview.png`](figures/figure_1_overview.png) | Final workflow overview panel |
| Figure 4A | [`source_data/Figure_4A_pixel_PR_curves_full.csv.gz`](source_data/Figure_4A_pixel_PR_curves_full.csv.gz) | Full pixel precision-recall coordinates for all six models |
| Figure 4D | [`figures/figure_4d_paired_iou_difference.png`](figures/figure_4d_paired_iou_difference.png) | Paired curriculum-minus-baseline mean IoU differences |
| Figure 6A | [`figures/figure_6a_object_recall_by_area.png`](figures/figure_6a_object_recall_by_area.png) | Object recall across reference-object area strata |
| Figure 6B | [`figures/figure_6b_iou_by_density.png`](figures/figure_6b_iou_by_density.png) | Image-level IoU across reference-count density strata |
| Figure S4 | [`figures/supplementary_figure_s4_model_interpretation.png`](figures/supplementary_figure_s4_model_interpretation.png) | Exploratory model-output interpretation |
| Main Table 1 | [`tables/main_table_1.csv`](tables/main_table_1.csv) | Pixel and boundary metrics |
| Main Table 2 | [`tables/main_table_2.csv`](tables/main_table_2.csv) | Object and count metrics |
| Supplementary Table S1 | [`tables/supplementary_table_s1.csv`](tables/supplementary_table_s1.csv) | Curriculum schedule |
| Supplementary Table S2 | [`tables/supplementary_table_s2.csv`](tables/supplementary_table_s2.csv) | Architecture and timing details |
| Supplementary Table S3 | [`tables/supplementary_table_s3.csv`](tables/supplementary_table_s3.csv) | Paired baseline-versus-curriculum statistics |

The other final plots are not deposited as publication images. Their
underlying CSV/NPZ data can be added here when exported from the original
analysis files; numerical claims should not be reconstructed by digitizing a
plot.

## Main Table 1: pixel and boundary performance

Values are image-level means with 95% percentile-bootstrap confidence intervals,
except finite HD95, which is the median with its 95% interval. All models used
the same 281 held-out images containing 2,799 reference objects. Masks used a
strict probability threshold of `> 0.5`; pixel AP used continuous probabilities.

| Architecture | Training | N images | N objects | Pixel IoU | Pixel Dice | Pixel AP | Finite HD95 (px) |
|---|---|---:|---:|---|---|---|---|
| U-Net | Baseline | 281 | 2,799 | 0.607 (0.587–0.626) | 0.741 (0.724–0.756) | 0.752 (0.735–0.768) | 71.09 (61.36–82.98) |
| U-Net | Curriculum | 281 | 2,799 | 0.634 (0.616–0.652) | 0.765 (0.751–0.778) | 0.719 (0.703–0.733) | 61.40 (52.65–74.07) |
| U-Net++ | Baseline | 281 | 2,799 | 0.644 (0.626–0.662) | 0.773 (0.759–0.786) | 0.679 (0.664–0.693) | 55.01 (49.68–61.71) |
| U-Net++ | Curriculum | 281 | 2,799 | 0.669 (0.653–0.687) | 0.793 (0.779–0.805) | 0.713 (0.698–0.727) | 42.25 (38.93–50.77) |
| TransUNet | Baseline | 281 | 2,799 | 0.603 (0.585–0.622) | 0.740 (0.725–0.755) | 0.723 (0.706–0.738) | 64.10 (57.04–76.52) |
| TransUNet | Curriculum | 281 | 2,799 | 0.623 (0.605–0.641) | 0.757 (0.743–0.770) | 0.733 (0.715–0.748) | 61.63 (53.44–73.82) |

## Main Table 2: object detection and counting

Values are image-level means with 95% percentile-bootstrap confidence intervals.
Count bias is predicted minus reference count; negative values indicate
undercounting. The intervals resample image tiles and are not well-level or
independent biological-replicate intervals.

| Architecture | Training | N images | N objects | Precision | Recall | Object F1 | Count MAE | Count bias |
|---|---|---:|---:|---|---|---|---|---|
| U-Net | Baseline | 281 | 2,799 | 0.855 (0.836–0.873) | 0.760 (0.735–0.783) | 0.783 (0.764–0.801) | 2.49 (2.17–2.85) | −1.48 (−1.89 to −1.07) |
| U-Net | Curriculum | 281 | 2,799 | 0.897 (0.880–0.914) | 0.784 (0.762–0.806) | 0.819 (0.802–0.836) | 2.23 (1.96–2.53) | −1.42 (−1.77 to −1.07) |
| U-Net++ | Baseline | 281 | 2,799 | 0.806 (0.786–0.826) | 0.842 (0.821–0.862) | 0.804 (0.787–0.821) | 2.12 (1.87–2.39) | 0.38 (0.02–0.74) |
| U-Net++ | Curriculum | 281 | 2,799 | 0.852 (0.834–0.869) | 0.862 (0.842–0.881) | 0.842 (0.826–0.857) | 1.90 (1.67–2.15) | 0.06 (−0.26 to 0.40) |
| TransUNet | Baseline | 281 | 2,799 | 0.789 (0.768–0.809) | 0.790 (0.767–0.812) | 0.769 (0.750–0.787) | 2.32 (2.04–2.63) | −0.36 (−0.76 to 0.03) |
| TransUNet | Curriculum | 281 | 2,799 | 0.771 (0.749–0.791) | 0.821 (0.798–0.842) | 0.777 (0.759–0.793) | 2.26 (2.01–2.53) | 0.27 (−0.09 to 0.65) |

## Supplementary Table S1: curriculum schedule

These are maximum epochs. Segmentation training used AdamW, batch size 16, and
cosine learning-rate decay. The expert-real development pool contained 1,629
training and 281 validation images; a separate 281-image set was held out for
testing. The 1,910 expert-labelled images shown for the final phase are the
complete development pool.

| Phase | Images used | Maximum epochs | Batch size | Initial learning rate | Principal loss |
|---|---:|---:|---:|---:|---|
| Diffusion-derived synthetic | 3,000 | 40 | 16 | 1 × 10⁻⁴ | Focal Tversky |
| Copy-paste synthetic | 1,500 | 80 | 16 | 5 × 10⁻⁵ | Focal Tversky |
| Pseudo-labelled real | 2,058 | 50 | 16 | 2 × 10⁻⁵ | Soft BCE; label smoothing 0.1 |
| Expert-labelled real | 1,910 | 200 | 16 | 1 × 10⁻⁵ | Focal Tversky |

## Supplementary Table S2: architectures and inference timing

Inputs were 512 × 512 RGB. Timing was measured per image with batch size 1 in
the same NVIDIA A100 environment for all three architectures.

| Architecture | Encoder | Input shape | Trainable parameters | Nontrainable parameters | Checkpoint (MB) | Seconds/image |
|---|---|---|---:|---:|---:|---:|
| U-Net | DenseNet121 | (512, 512, 3) | 22,254,369 | 81,600 | 256.307991 | 0.095627 |
| U-Net++ | ResNet50 | (512, 512, 3) | 7,567,875 | 1,469,312 | 103.956306 | 0.089694 |
| TransUNet | EfficientNetB0 | (512, 512, 3) | 41,346,499 | 4,007,555 | 489.356514 | 0.086817 |

## Supplementary Table S3: paired comparisons

Differences are curriculum minus baseline; negative count-MAE differences favor
curriculum. Tests use 281 paired image tiles, the Pratt convention for Wilcoxon
signed-rank tests, and Holm correction across architectures within each metric.

| Architecture | Metric | Baseline mean | Curriculum mean | Mean paired difference (95% CI) | Wilcoxon W | Unadjusted p | Holm-adjusted p |
|---|---|---:|---:|---|---:|---:|---:|
| U-Net | IoU | 0.606555 | 0.634031 | 0.027475 (0.019320 to 0.035162) | 10117.0 | 2.116263e-12 | 4.232525e-12 |
| U-Net | Dice | 0.741148 | 0.764753 | 0.023605 (0.016468 to 0.030624) | 10194.0 | 3.166892e-12 | 6.333785e-12 |
| U-Net | Object F1 | 0.781585 | 0.816832 | 0.035248 (0.022439 to 0.047548) | 10953.0 | 6.705925e-08 | 1.341185e-07 |
| U-Net | Count MAE | 2.498221 | 2.238434 | −0.259786 (−0.419929 to −0.088968) | 14105.5 | 1.957327e-02 | 5.815857e-02 |
| U-Net++ | IoU | 0.644454 | 0.669430 | 0.024975 (0.017900 to 0.032189) | 9618.0 | 1.439026e-13 | 4.317079e-13 |
| U-Net++ | Dice | 0.773237 | 0.792616 | 0.019379 (0.013639 to 0.025170) | 9652.0 | 1.735537e-13 | 5.206612e-13 |
| U-Net++ | Object F1 | 0.802967 | 0.840948 | 0.037980 (0.025549 to 0.050674) | 10445.0 | 2.576613e-09 | 7.729840e-09 |
| U-Net++ | Count MAE | 2.113879 | 1.925267 | −0.188612 (−0.373843 to −0.003559) | 13689.5 | 1.938619e-02 | 5.815857e-02 |
| TransUNet | IoU | 0.603296 | 0.623292 | 0.019996 (0.011644 to 0.030142) | 11819.0 | 7.229151e-09 | 7.229151e-09 |
| TransUNet | Dice | 0.739851 | 0.756919 | 0.017069 (0.009320 to 0.026827) | 11897.0 | 1.014468e-08 | 1.014468e-08 |
| TransUNet | Object F1 | 0.767775 | 0.776429 | 0.008654 (−0.005147 to 0.022040) | 17314.5 | 3.677161e-01 | 3.677161e-01 |
| TransUNet | Count MAE | 2.327402 | 2.266904 | −0.060498 (−0.249110 to 0.145996) | 17517.0 | 8.800840e-01 | 8.800840e-01 |
