# Model checkpoints

Model weight files are excluded from Git because they are large binary research artifacts. For a reproducible release:

1. Deposit the six final checkpoints (three architectures × baseline/curriculum), relevant U-Net++ phase checkpoints, and the pseudo-label teacher checkpoint in a versioned archive.
2. Record the exact archive filename, download URL/DOI, SHA-256 digest, architecture, and training regime in `manifests/model_registry.csv`.
3. Verify each checkpoint against the reported test outputs before release.

**Known provenance issue to resolve:** historical local filenames may not uniquely distinguish baseline and curriculum weights, and earlier evaluation registries may have reversed labels. TransUNet exports also reuse a stage-3-style filename. Do not infer model identity from a filename alone.
