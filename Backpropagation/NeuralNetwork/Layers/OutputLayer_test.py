import numpy as np
from NeuralNetwork.Layers import OutputLayer


def test_returns_input_without_softmax():
    data = np.array([[1, 2, 3], [2, 3, 4]])
    output = OutputLayer(input_neurons=3).forward_pass(data)

    assert np.all(np.isclose(output, data))


def test_applies_softmax():
    data = np.array([1, 2, 3])
    output = OutputLayer(input_neurons=3, softmax=True).forward_pass(data)
    expected = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.all(np.isclose(output, expected))


def test_backward_pass_no_softmax():
    data = np.array([[1, 2, 3, 6, 4], [1, 2, 6, 3, 5]])
    output = OutputLayer(input_neurons=5, softmax=False).backward_pass(
        data, None, None
    )[1]
    assert np.all(np.isclose(output, data))
