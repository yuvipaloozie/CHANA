# Legacy notebook exports

The files in `python_exports/` are the supplied Colab-to-Python exports, except that `++` was normalized to `pp` in two filenames for portability. Some contain Colab magics, repeated cells, Google Drive paths, and environment-install commands.

Use them in Colab after setting the data paths. Reusable local inference code is under `src/chana/`.

| Export | Purpose |
|---|---|
| `chana_diffusionv2.py` | Diffusion-named/generative domain workflow supplied for Phase 1 |
| `chana_copy_paste_gen.py` | Copy-paste domain construction |
| `chana_pseudo_labelling.py` | Fixed TransUNet teacher inference and pseudo-label construction |
| `chana_preprocessing_rgb.py` | Original V9 RGB enhancement and preprocessing analysis |
| `*_with_domains.py` | Sequential curriculum training by architecture |
| `*_without_domains.py` | Expert-real-only baseline training by architecture |
| `chana_inference_notebook.py` | Historical inference workflow |
