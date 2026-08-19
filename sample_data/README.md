# Sample data

`public_example/` contains one cleared 512 x 512 image, expert mask,
probability map, binary prediction, watershed labels, and overlay. Validate it
with:

```bash
python scripts/validate_sample_data.py
```

The derived files are internally consistent; exact originating-checkpoint
linkage remains pending. `ten_image_example/` records a deterministic ten-pair
sample staged locally from the locked bundle. Its image and mask directories
remain ignored until redistribution clearance is confirmed.
