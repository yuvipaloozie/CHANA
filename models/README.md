# Model checkpoints

The six primary checkpoints are stored here with Git LFS under logical names.
`manifests/model_registry.csv` records their model IDs, sizes, SHA-256 hashes,
and historical aliases. Verify their bytes, hashes, and architecture
compatibility with:

```bash
python scripts/validate_checkpoints.py --weights-dir path/to/checkpoints
```

For all three architectures, the historical `Domain` files are baseline and
the historical `no_Domain` files are curriculum. The canonical files remove
that ambiguity; model IDs and hashes remain authoritative. Additional phase
checkpoints and the pseudo-label teacher are not included yet.
