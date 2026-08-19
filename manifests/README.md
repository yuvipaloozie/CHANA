# Manifests

The model registry identifies the six checkpoints by model ID, file size, and
SHA-256. The data, split, and domain manifests are templates that must be
populated from the final dataset inventory.

- `dataset_manifest.csv`: one row per image-mask pair across all domains.
- `split_manifest.csv`: the authoritative train/validation/test membership and scan grouping.
- `domain_manifest.csv`: source of generated, composited, pseudo-labeled, or expert labels.
- `model_registry.csv`: one row per released checkpoint, including SHA-256 digest.

Run `python scripts/validate_manifests.py --require-populated` after adding the
final rows. The validator checks study counts, unique IDs, and scan overlap.

Do not publish direct local Drive paths, personal identifiers, or restricted acquisition metadata.
