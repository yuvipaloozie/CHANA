# Ten-image inference sample

Ten image-mask pairs are staged locally for multi-image inference testing. They
were selected from the 281-row locked manifest with Python 3.10 using:

```python
sorted(random.Random(20260819).sample(range(281), 10))
```

`selection_manifest.csv` records the selected 1-based CSV rows, public sample
IDs, and expected SHA-256 hashes. The local files are renamed
`sample_001.png` through `sample_010.png`.

The image and mask directories are ignored from Git while redistribution
clearance is pending. After clearance, remove the two matching `.gitignore`
rules, change `clearance_status` to `cleared`, and run inference against the
registered checkpoint(s). Do not publish the original locked-test filenames.
