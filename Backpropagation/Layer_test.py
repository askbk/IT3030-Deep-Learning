from Layer import Layer
import numpy as np


def test_layer_constructor():
    Layer()
    Layer(
        neurons=5,
        activation_function="sigmoid",
        softmax=False,
        initial_weight_range=(-0.1, 0.1),
    )


def test_forward_pass():
    layer = Layer(neurons=1, input_neurons=1)
    data = np.array([[0], [1]])

    output = layer.forward_pass(data)
    assert not any(np.isnan(output))


def test_forward_pass_correct_output():
    layer = Layer(
        input_neurons=1,
        weights=np.array([[1]]),
        activation_function="sigmoid",
        softmax=False,
    )
    actual = layer.forward_pass(np.array([[-1], [0], [1]]))
    expected = [[0.26894142], [0.5], [0.73105858]]
    assert all([np.isclose(a, b) for a, b in zip(actual, expected)])

    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1, 0.2]]),
        activation_function="sigmoid",
        softmax=False,
    )
    actual = layer.forward_pass(np.array([[-1, 1], [0, -1], [1, 0]]))
    expected = [[0.52498], [0.45017], [0.52498]]
    print(actual)
    assert all(
        [np.isclose(a, b) for x, y in zip(actual, expected) for a, b in zip(x, y)]
    )
