from Layer import Layer
import numpy as np
import pytest


def test_layer_constructor():
    Layer(neurons=1, input_neurons=5)
    Layer(
        input_neurons=55,
        neurons=5,
        activation_function="sigmoid",
        initial_weight_range=(-0.1, 0.1),
    )


def test_layer_correct_weight_bias_dimensions():
    with pytest.raises(ValueError):
        Layer(input_neurons=3, neurons=2, weights=np.array([[1, 2], [1, 2]]))

    with pytest.raises(ValueError):
        Layer(input_neurons=3, neurons=2, weights=np.array([[1, 2, 3], [1, 2, 3]]))

    with pytest.raises(ValueError):
        Layer(input_neurons=1, neurons=2, bias_weights=np.array([1, 2]))

    with pytest.raises(ValueError):
        Layer(input_neurons=1, neurons=2, bias_weights=np.array([[1, 2], [1, 2]]))


def test_forward_pass():
    layer = Layer(neurons=1, input_neurons=1)
    data = np.array([[0], [1]])
    output = layer.forward_pass(data)

    assert not np.any(np.isnan(output))


def test_forward_pass_correct_output_base_case():
    layer = Layer(
        input_neurons=1,
        neurons=1,
        weights=np.array([[1]]),
        activation_function="sigmoid",
        bias=False,
    )
    actual = layer.forward_pass(np.array([[-1], [0], [1]]))
    expected = np.array([[0.26894142], [0.5], [0.73105858]])

    assert np.all(np.isclose(actual, expected))


def test_forward_pass_correct_output_multiple_input():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="sigmoid",
        bias=False,
    )
    data = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = layer.forward_pass(data)
    expected = [[0.52498], [0.45017], [0.52498]]

    assert np.all(np.isclose(actual, expected))


def test_forward_pass_correct_output_with_bias():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="sigmoid",
        bias=True,
        bias_weights=np.array([[0.1]]),
    )
    data = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = layer.forward_pass(data)
    expected = np.array([[0.549834], [0.475021], [0.549834]])

    assert np.all(np.isclose(actual, expected))