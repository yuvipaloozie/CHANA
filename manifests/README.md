# Model registry

`model_registry.csv` is retained because it is the authoritative mapping from
each semantic model ID to its canonical checkpoint, historically reversed
filename, byte size, and SHA-256 hash.

Empty dataset/domain/split templates were removed. The exact image-to-split and
source-scan mapping remains an explicit release requirement in the main README.
