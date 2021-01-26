from InputLayer import InputLayer
import numpy as np


def test_forward_pass_returns_input_with_intercept():
    data = np.array([1, 0, 1])
    expected = np.array([1, 1, 0, 1])
    output = InputLayer().forward_pass(data)
    assert np.all(np.isclose(expected, output))