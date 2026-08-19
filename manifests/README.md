# Manifests

These CSV files are intentionally header-only in the scaffold. The publication release must populate them with stable, nonidentifying IDs and relative or archive-resolvable paths.

- `dataset_manifest.csv`: one row per image-mask pair across all domains.
- `split_manifest.csv`: the authoritative train/validation/test membership and scan grouping.
- `domain_manifest.csv`: provenance of generated, composited, pseudo-labeled, or expert labels.
- `model_registry.csv`: one row per released checkpoint, including SHA-256 digest.

Run `python scripts/validate_manifests.py --require-populated` before creating the archival release. The validator enforces the reported study counts and unique image identifiers.

Do not publish direct local Drive paths, personal identifiers, or restricted acquisition metadata.
