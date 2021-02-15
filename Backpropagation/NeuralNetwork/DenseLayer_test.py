import pytest
import numpy as np
from NeuralNetwork.DenseLayer import DenseLayer


def test_layer_constructor():
    DenseLayer(neurons=1, input_neurons=5)
    DenseLayer(
        input_neurons=55,
        neurons=5,
        activation_function="sigmoid",
        initial_weight_range=(-0.1, 0.1),
    )


def test_layer_correct_weight_bias_dimensions():
    with pytest.raises(ValueError):
        DenseLayer(neurons=2, input_neurons=2, weights=np.array([[1, 2], [1, 2]]))

    with pytest.raises(ValueError):
        DenseLayer(neurons=2, input_neurons=2, weights=np.array([[1, 2, 3], [1, 2, 3]]))


def test_forward_pass():
    layer = DenseLayer(neurons=1, input_neurons=2)
    data = np.array([0, 1])
    output = layer.forward_pass(data)

    assert not np.any(np.isnan(output))


def test_forward_pass_correct_output_base_case():
    layer = DenseLayer(
        input_neurons=1,
        neurons=1,
        weights=np.array([[1]]),
        activation_function="sigmoid",
        use_bias=False,
    )
    cases = [np.array([-1]), np.array([0]), np.array([1])]
    actual = [layer.forward_pass(case) for case in cases]
    expected = [[0.26894142], [0.5], [0.73105858]]

    assert np.all(np.isclose(actual, expected))


def test_forward_pass_correct_output_multiple_input():
    layer = DenseLayer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.2]]),
        activation_function="sigmoid",
        use_bias=False,
    )
    cases = [np.array([-1, 1]), np.array([0, -1]), np.array([1, 0])]
    actual = [layer.forward_pass(case) for case in cases]
    expected = [[0.52498], [0.45017], [0.52498]]

    assert np.all(np.isclose(actual, expected))


def test_forward_pass_correct_output_with_bias():
    layer = DenseLayer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.1], [0.2]]),
        activation_function="sigmoid",
    )
    cases = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = [layer.forward_pass(case) for case in cases]
    expected = np.array([[0.549834], [0.475021], [0.549834]])

    assert np.all(np.isclose(actual, expected))


def test_hyperbolic_tangent_activation():
    layer = DenseLayer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.1], [0.2]]),
        activation_function="tanh",
        use_bias=True,
    )
    cases = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = [layer.forward_pass(case) for case in cases]
    expected = np.array([[0.19737532], [-0.09966799], [0.19737532]])

    assert np.all(np.isclose(actual, expected))


def test_relu_activation():
    layer = DenseLayer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.1], [0.2]]),
        activation_function="relu",
        use_bias=True,
    )
    cases = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = [layer.forward_pass(case) for case in cases]
    expected = np.array([[0.2], [0], [0.2]])
    assert np.all(np.isclose(actual, expected))


def test_linear_activation():
    layer = DenseLayer(
        input_neurons=2,
        neurons=1,
        weights=np.array([[0.1], [0.1], [0.2]]),
        activation_function="linear",
        use_bias=True,
    )
    cases = np.array([[-1, 1], [0, -1], [1, 0]])
    actual = [layer.forward_pass(case) for case in cases]
    expected = np.array([[0.2], [-0.1], [0.2]])
    assert np.all(np.isclose(actual, expected))


def test_backward_pass_updates_weights():
    Y = np.array([-0.1, 0.2])
    Y_with_bias = np.concatenate(([1], Y))
    W = np.array([[0.1, 0.2, 0.3], [0.3, -0.1, -0.2], [0.2, 0.5, 0.1]])
    YW = Y_with_bias @ W
    f = lambda X: 1 / (1 + np.exp(-X))
    Z = f(YW)
    layer = DenseLayer(
        input_neurons=2,
        neurons=3,
        weights=W,
        activation_function="sigmoid",
        use_bias=True,
    )

    _weight_jacobian, _prev_layer_jacobian = layer.backward_pass(Z, Z, Y)