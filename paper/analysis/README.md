# Manuscript analysis

`legacy/` contains the final supplied Python exports used to calculate metrics and generate main/supplemental manuscript assets. They are retained as computational provenance but still contain Colab-specific paths and assume the private analysis directory structure.

Before publication, refactor the final locked analysis into a manifest-driven command that reads the six verified model checkpoints and writes the source-data files listed in `paper/source_data/README.md`. Keep numerical source data separate from aesthetic GraphPad Prism project files.
