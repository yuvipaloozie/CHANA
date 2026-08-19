import numpy as np

from chana.preprocessing import normalize_preprocessed_rgb, preprocess_v9


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


def test_preprocessed_rgb_normalization_does_not_repeat_v9_enhancement():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    normalized = normalize_preprocessed_rgb(bgr, size=(4, 4), input_order="bgr")

    expected_rgb = np.zeros((4, 4, 3), dtype=np.float32)
    expected_rgb[..., 2] = 1.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    np.testing.assert_allclose(normalized, (expected_rgb - mean) / std)
