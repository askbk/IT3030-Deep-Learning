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
        Layer(input_neurons=1, neurons=2, bias_weights=np.array([[1, 2]]))

    with pytest.raises(ValueError):
        Layer(input_neurons=1, neurons=2, bias_weights=np.array([[1, 2], [1, 2]]))


def test_forward_pass():
    layer = Layer(neurons=1, input_neurons=2)
    data = np.array([0, 1])
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
    cases = [np.array([-1]), np.array([0]), np.array([1])]
    actual = [layer.forward_pass(case) for case in cases]
    expected = [[0.26894142], [0.5], [0.73105858]]

    assert np.all(np.isclose(actual, expected))


def test_forward_pass_correct_output_multiple_input():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="sigmoid",
        bias=False,
    )
    cases = [np.array([-1, 1]), np.array([0, -1]), np.array([1, 0])]
    actual = [layer.forward_pass(case) for case in cases]
    expected = [[0.52498], [0.45017], [0.52498]]

    assert np.all(np.isclose(actual, expected))


def test_forward_pass_correct_output_with_bias():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="sigmoid",
        bias=True,
        bias_weights=np.array([0.1]),
    )
    cases = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = [layer.forward_pass(case) for case in cases]
    expected = np.array([[0.549834], [0.475021], [0.549834]])

    assert np.all(np.isclose(actual, expected))


def test_hyperbolic_tangent_activation():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="tanh",
        bias=True,
        bias_weights=np.array([0.1]),
    )
    data = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = layer.forward_pass(data)
    expected = np.array([[0.19737532], [-0.09966799], [0.19737532]])

    assert np.all(np.isclose(actual, expected))


def test_relu_activation():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="relu",
        bias=True,
        bias_weights=np.array([0.1]),
    )
    data = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = layer.forward_pass(data)
    expected = np.array([[0.2], [0], [0.2]])
    assert np.all(np.isclose(actual, expected))


def test_linear_activation():
    layer = Layer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="linear",
        bias=True,
        bias_weights=np.array([0.1]),
    )
    data = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = layer.forward_pass(data)
    expected = np.array([[0.2], [-0.1], [0.2]])
    assert np.all(np.isclose(actual, expected))


def test_backward_pass_updates_weights():
    Y = np.array([-0.1, 0.2])
    W = np.array([[0.3, -0.1, -0.2], [0.2, 0.5, 0.1]])
    YW = Y @ W
    f = lambda X: 1 / (1 + np.exp(-X))
    Z = f(YW)
    bias = np.array([0.1, 0.2, 0.3])
    layer = Layer(
        input_neurons=2,
        neurons=3,
        weights=W,
        activation_function="sigmoid",
        bias_weights=bias,
    )

    updated_layer, _ = layer.backward_pass(Z, Z, Y, 0.1)

    new_Z = updated_layer.forward_pass(Y)
    assert not np.all(np.isclose(new_Z, Z))
    assert not np.all(np.isclose(updated_layer._bias, bias))