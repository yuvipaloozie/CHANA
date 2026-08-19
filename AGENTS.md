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
- Treat the final Word figure/table documents as the authority for panel selection and publication-target numerical values. Older notebook composites contain extra panels that were intentionally removed. A displayed value is still not verified machine-readable provenance: preserve it, transcribe it explicitly, and mark its source CSV pending when necessary.

## Editing and implementation rules

- Preserve historical notebook/Python exports as provenance; place corrected reusable code under `src/chana/` or new manifest-driven scripts.
- Do not silently revise manuscript numbers. Preserve the final Word values as publication targets and trace every reported value to machine-readable source data and a verified model ID before calling it reproducible.
- Keep the locked test set outside threshold, checkpoint, architecture, and figure-driven model selection.
- Use stable image/model identifiers and SHA-256 hashes in public outputs.
- Keep local Google Drive paths, unregistered weights, raw data, credentials, and restricted metadata out of Git. The six primary registered checkpoints are stored with Git LFS.
- Add tests for reusable code changes and run both validators plus `pytest` before committing.
- Use Python 3.10 and the checkpoint-compatible TensorFlow 2.16.2/Keras 3.12.4 inference environment. The exact historical training environment remains unresolved because the Colab exports installed unpinned TensorFlow.

## Validation commands

```bash
python scripts/validate_manifests.py
python scripts/validate_source_data.py
python -m pytest
```

## Release blockers

Do not describe the repository as fully reproducible until the exact split manifest, reported-result checkpoint identities, remaining figure/table source data, exact checkpoint linkage for the public sample, final license, and DOI archive are present.
