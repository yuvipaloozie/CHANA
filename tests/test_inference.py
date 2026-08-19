import numpy as np

from chana.inference import select_probability_output


def test_selects_unetpp_final_output_and_transunet_first_output():
    outputs = [np.full((1, 4, 4, 1), value, dtype=np.float32) for value in (1, 2, 3, 4)]
    assert np.all(select_probability_output(outputs, "unetpp") == 4)
    assert np.all(select_probability_output(outputs[:3], "transunet") == 1)


def test_selects_single_output():
    output = np.ones((2, 4, 4, 1), dtype=np.float32)
    assert select_probability_output(output, "unet").shape == (2, 4, 4)
