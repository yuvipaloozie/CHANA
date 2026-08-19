# CHANA model card

## Intended use

CHANA segments expert-defined osteoclast regions in TRAP-stained cultured-cell bright-field images and derives analytical object counts and morphometric measurements after watershed separation. It is intended for research-method development and retrospective analysis within settings similar to the study data.

## Not intended for

- clinical diagnosis, prognosis, or treatment decisions;
- independent verification of the ≥3-nuclei biological criterion;
- claims about drug efficacy or prospective screening performance;
- direct use on other stains, microscopes, species, cell lines, or tissue sections without external validation.

## Model families

Six primary models compare U-Net, U-Net++, and TransUNet under expert-real-only baseline training and sequential curriculum training. The curriculum phases are diffusion-derived, copy-paste, pseudo-labeled real, and expert-labeled real.

## Output interpretation

The neural network produces a semantic probability map. A threshold of 0.5 yields a binary foreground mask. Filled foreground regions are separated using distance-transform watershed with a 20-pixel peak spacing, then filtered to areas from 50 through 10,000 pixels, inclusive. These are algorithmic object instances, not manually supplied instance identities.

## Evaluation

Reported endpoints include pixel precision-recall, IoU, Dice, HD95, centroid-matched object precision/recall/F1 with a 25-pixel gate, count error/agreement, and robustness by object size, field density, and foreground fraction. The fixed expert-labeled test set contains 281 images.

## Limitations and risks

The dataset comes from a single study context and may encode stain, acquisition, or batch-specific features. A random split may underestimate distribution shift if scan grouping is not verified. Dense or overlapping regions can merge, fragment, or generate watershed artifacts. Small objects are more difficult to recall. Probability calibration, external-site generalization, and biological outcome validity require additional study.

## Provenance requirements

Do not release or deploy a checkpoint until it has a unique model ID, architecture/training-regime label, SHA-256 digest, source code commit, data-manifest checksum, and expected evaluation output. Resolve the known historical filename ambiguity before publication.
