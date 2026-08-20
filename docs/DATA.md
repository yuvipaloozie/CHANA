# Data summary

| Domain or split | Images | Label source |
|---|---:|---|
| Diffusion-derived | 3,000 | Generated structural masks |
| Copy-paste | 1,500 | Masks transformed with composited objects |
| Pseudo-labelled real | 2,058 | Fixed TransUNet teacher |
| Expert-real training | 1,629 | Expert semantic masks |
| Expert-real validation | 281 | Expert semantic masks |
| Expert-real test | 281 | Expert semantic masks |

The expert-real total is 2,191 image-mask pairs. Curriculum-domain exposures
must not be added to that total as if they were independently collected
biological samples.

## Split limitation

The exact image-to-split list and its source-scan identifiers have not yet been
deposited. Historical exports alone do not establish the final membership or
group independence. Until the original saved lists or data folders are checked,
describe the evaluation set only as a fixed 281-image holdout—not as scan-,
well-, or biological-replicate independent.

For the archival release, add one CSV containing stable image ID, relative image
and mask paths, split, and source-scan ID. Check that no source scan occurs in
both development and test sets before making any grouped-split claim.
