# External-method benchmark

This folder contains the compact, machine-readable outputs from applying three
external methods to the same 281-image CHANA test set. The `YOLO` label in the
CSV files denotes **NOISe-MH**, the osteoclast instance-segmentation model built
on YOLOv8, not a generic YOLO baseline.

The evaluated external methods were Cellpose, CellProfiler 4.2.8, and NOISe-MH.
There was no separate threshold-only method. The CellProfiler pipeline uses the
inverted green channel, global Otsu thresholding, shape-based declumping, and a
minimum retained area of 400 pixels.

Recalculate the aggregate point estimates with:

```bash
python scripts/summarize_external_benchmark.py \
  benchmarks/external_methods/results/per_image_metrics.csv \
  --output benchmarks/external_methods/results/model_summary.csv
```

Pixel IoU and Dice use binary masks. HD95 is the larger of the two directed
95th-percentile foreground-boundary distances; a one-sided empty mask is
assigned 500 pixels and a pair of empty masks is assigned zero. Object results
use summed TP, FP, and FN counts. `Count_R2` is linear-fit R-squared (Pearson
correlation squared), not prediction-agreement `r2_score`.

The input images, masks, overlays, and generated TIFF/PNG outputs are not stored
in GitHub. They should be obtained from the associated data archive. NOISe code
should be obtained from upstream commit
`376290a4a61b2f2b0031b9b0bcaa65f2edbe7ef3`; the tested `noise_mh.pt` checkpoint
had SHA-256
`da7b42870cfe4dc2519b320d4b844ff8096e358367f4c21f054ccb775ca861b0`.

Exact Cellpose inference settings and the original bootstrap script were not
present in the recovered analysis folder. Accordingly, the deposited CSVs
support inspection of the completed comparison, but those two stages should not
be described as independently rerunnable until their original configuration is
recovered.
