# Data and manifest specification

## Reported study inventory

| Domain/split | Images | Label source |
|---|---:|---|
| Diffusion-derived | 3,000 | Generated structural masks |
| Copy-paste | 1,500 | Masks transformed with composited objects |
| Pseudo-labeled real | 2,058 | Fixed expert-real-fine-tuned TransUNet teacher |
| Expert-real training | 1,629 | Expert semantic masks |
| Expert-real validation | 281 | Expert semantic masks |
| Expert-real test | 281 | Expert semantic masks |

The expert-real total is 2,191 image-mask pairs. Curriculum-domain counts are exposures and must not be added to the expert-real total as if all observations were independently collected biological samples.

## Required identifiers

Every released record must receive a stable `image_id`. Expert-real images should also receive a `scan_id` so the stated original-scan independence of the test set can be checked. Synthetic and pseudo-labeled records require a `source_id` that links them to the generating image, background, object bank, teacher checkpoint, or seed as applicable.

The final manifests, not directory enumeration order, are authoritative. Paths should be relative to the deposited archive or resolvable URLs. Do not publish local Google Drive paths.

## Split caution

The manuscript specifies 1,629/281/281 expert-real train/validation/test images and a random split with test independence at the original-scan level. Historical exports include fractional `train_test_split` calls with seeds 999 and 42 that do not, by themselves, reconcile with those reported counts. Before release:

1. reconstruct the exact membership from the actual data folders or saved lists;
2. populate `split_manifest.csv` with stable IDs and scan IDs;
3. verify there is no scan overlap between test and development sets;
4. freeze the manifest and checksum it;
5. use only that manifest for final reported evaluation.

## Public release strategy

GitHub should contain code, configuration, schemas, and a redistributable smoke-test example. A DOI-backed archive should contain the public image/mask data or an access-controlled data statement, source data underlying all graphs and tables, verified checkpoints, and checksums. The manuscript Data Availability statement should point to the permanent record.
