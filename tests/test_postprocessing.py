import numpy as np

from chana.postprocessing import filter_and_measure, measurements_frame, separate_objects


def test_empty_mask_produces_no_objects():
    labels = separate_objects(np.zeros((64, 64), dtype=np.uint8))
    filtered, measurements = filter_and_measure(labels)
    assert labels.max() == 0
    assert filtered.max() == 0
    assert measurements_frame(measurements).empty


def test_two_components_are_measured():
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[10:25, 10:25] = 1
    mask[55:75, 60:80] = 1
    labels = separate_objects(mask, min_peak_distance=5)
    filtered, measurements = filter_and_measure(labels, min_area=50, max_area=1000)
    assert filtered.max() == 2
    assert len(measurements) == 2
    assert all(item.area_px > 50 for item in measurements)
