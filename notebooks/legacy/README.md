# Legacy notebook exports

The files in `python_exports/` are unmodified provenance copies of the supplied Colab-to-Python exports, except that `++` was normalized to `pp` in two filenames for portability. They preserve the historical workflow but are not package-quality scripts: some contain Colab magics, repeated cell definitions, hard-coded Google Drive paths, and environment installation commands.

Use these files to audit the original implementation and compare it with the reusable modules under `src/chana/`. Do not assume a legacy export runs from top to bottom outside Colab without review.

| Export | Purpose |
|---|---|
| `chana_diffusionv2.py` | Diffusion-named/generative domain workflow supplied for Phase 1 |
| `chana_copy_paste_gen.py` | Copy-paste domain construction |
| `chana_pseudo_labelling.py` | Fixed TransUNet teacher inference and pseudo-label construction |
| `*_with_domains.py` | Sequential curriculum training by architecture |
| `*_without_domains.py` | Expert-real-only baseline training by architecture |
| `chana_inference_notebook.py` | Historical inference workflow |

The final publication release should preserve these files while making the manifest-driven workflows the recommended entry points.
