# Model checkpoints

The six primary evaluation checkpoints and the fixed TransUNet pseudo-label
teacher are stored here with Git LFS under logical names.
`manifests/model_registry.csv` records their model IDs, sizes, SHA-256 hashes,
and historical aliases. Verify their bytes and hashes with:

```bash
python scripts/validate_checkpoints.py --weights-dir path/to/checkpoints
```

For all three architectures, the historical `Domain` files are baseline and
the historical `no_Domain` files are curriculum. The canonical files remove
that ambiguity; model IDs and hashes remain authoritative. The teacher is
`transunet_pseudolabel_teacher.weights.h5` and is used only by the supplied
pseudo-labelling export, not as a seventh evaluation model.
