# Sample data

`public_example/` contains one cleared 512 x 512 image, expert mask,
probability map, binary prediction, watershed labels, and overlay. Validate it
with:

```bash
python scripts/validate_sample_data.py
```

The files are internally consistent. The supplied probability map is not tied
to an exact registered checkpoint, so use the input image—not the supplied
probability map—as the example for fresh model inference.

`public_example/pairs.csv` provides the image-mask pair to
`scripts/compare_models.py`. Add only redistribution-cleared pairs to that file.
