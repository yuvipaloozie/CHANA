import numpy as np

from chana.preprocessing import preprocess_v9


def test_preprocess_v9_is_deterministic_and_finite():
    image = np.zeros((37, 41, 3), dtype=np.uint8)
    image[10:25, 12:30] = (40, 120, 220)
    first = preprocess_v9(image, size=(64, 64), input_order="bgr")
    second = preprocess_v9(image, size=(64, 64), input_order="bgr")
    assert first.shape == (64, 64, 3)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_preprocess_v9_rejects_non_rgb_shape():
    with np.testing.assert_raises(ValueError):
        preprocess_v9(np.zeros((16, 16), dtype=np.uint8))
