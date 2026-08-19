# Public-example validation record

Validation performed on 2026-08-19 with the six author-supplied TIFFs and the
two registered U-Net++ checkpoints.

## Internally verified source bundle

- All six TIFFs are spatially aligned at 512 x 512 pixels and match the
  SHA-256 values in `manifest.yml`.
- The expert reference mask is binary (`0`, `255`) with 15,395 foreground
  pixels.
- The probability map is float32 with values from approximately 2.07e-7 to
  1.0.
- The supplied binary prediction equals the probability map thresholded at
  `>= 0.5` and hole-filled, with zero differing pixels. No probability value is
  exactly 0.5, so `>` and `>=` produce the same mask for this example.
- The supplied watershed labels contain 13 objects. Current code reproduces
  the partition exactly; only the numeric label ordering differs.
- The overlay reconstructs pixel-for-pixel from the deposited image, expert
  mask, binary prediction, and documented cyan/pink contour colors.
- The author confirmed that this paper example may be included in the public
  repository.

## Exact checkpoint identity remains unresolved

The author identified the source inference as domain-curriculum U-Net++.
However, the deposited probability map is not reproduced exactly by either
registered U-Net++ checkpoint under the verified TensorFlow 2.16.2/Keras
3.12.4 inference environment:

| Registered checkpoint | Threshold-mask IoU with deposited output | Differing pixels | Probability MAE | Pearson r |
|---|---:|---:|---:|---:|
| `unetpp_baseline` (`UNetPlusPlus_Domain.weights.h5`) | 0.728679 | 4,263 | 0.016273 | 0.845077 |
| `unetpp_curriculum` (`Unetplusplus_no_Domain.weights.h5`) | 0.854472 | 2,257 | 0.008568 | 0.922399 |

These diagnostics are not manuscript performance claims. The earlier
closer-baseline interpretation resulted from reading the reversed legacy
filenames literally. Hash-linked evaluation and phase-checkpoint source data
identify the closer file as the curriculum checkpoint. Neither comparison is
exact, so proximity still does not establish the originating checkpoint. Until
the exact originating checkpoint
or matching evaluation registry is recovered, this bundle is suitable for
testing file parsing, thresholding, watershed partitioning, and overlay
construction, but not for asserting an exact checkpoint-to-probability
regression.
