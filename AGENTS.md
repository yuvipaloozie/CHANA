# CHANA agent instructions

Read `docs/HANDOFF_CONTEXT.md` before changing code, data documentation, manuscript-support files, or reported numbers.

## Project objective

Prepare CHANA as a rigorous, public reproducibility repository supporting a planned PLOS Computational Biology manuscript on sequential domain-curriculum learning for semantic segmentation and morphometric quantification of cultured osteoclasts.

## Non-negotiable scientific constraints

- The model segments expert-defined osteoclast regions; it does not independently count nuclei.
- Do not claim prospective clinical utility, drug efficacy, or validated drug-screening performance. No such experiment is available.
- The expert-real inventory is 2,191 image-mask pairs: 1,629 training, 281 validation, and 281 test.
- Curriculum-domain sizes are 3,000 diffusion-derived, 1,500 copy-paste, 2,058 pseudo-labeled, and 1,629 expert-real training images.
- The test set is a random holdout claimed to be independent at the original-scan level, not a well-level split. This claim must remain qualified until a file-level manifest with `scan_id` is populated and verified.
- Never infer checkpoint identity from a filename alone. Historical baseline/curriculum checkpoint naming is ambiguous.
- Treat the final Word figure/table documents as the authority for panel selection. Older notebook composites contain extra panels that were intentionally removed.

## Editing and implementation rules

- Preserve historical notebook/Python exports as provenance; place corrected reusable code under `src/chana/` or new manifest-driven scripts.
- Do not silently revise manuscript numbers. Trace every reported value to machine-readable source data and a verified model ID.
- Keep the locked test set outside threshold, checkpoint, architecture, and figure-driven model selection.
- Use stable image/model identifiers and SHA-256 hashes in public outputs.
- Keep local Google Drive paths, weights, raw data, credentials, and restricted metadata out of Git.
- Add tests for reusable code changes and run both validators plus `pytest` before committing.
- Use Python 3.10 for the pinned TensorFlow 2.15 environment; NumPy must remain below 2 for that stack.

## Validation commands

```bash
python scripts/validate_manifests.py
python scripts/validate_source_data.py
python -m pytest
```

## Release blockers

Do not describe the repository as fully reproducible until the exact split manifest, verified checkpoint registry/hashes, remaining figure/table source data, public smoke-test example, final license, and DOI archive are present.
