from Network import Network
from Layer import Layer
from InputLayer import InputLayer
import numpy as np


def test_network_constructor():
    Network(
        layers=[], loss_function="mse", regularization=None, regularization_rate=0.001
    )


def test_network_forward_pass_base_case():
    network = Network(
        layers=[
            Layer(
                input_neurons=1,
                neurons=1,
                activation_function="sigmoid",
                softmax=False,
                weights=np.array([[1]]),
                bias=True,
                bias_weights=np.array([[1]]),
            )
        ],
        loss_function="mse",
        regularization=None,
    )

    actual = network.forward_pass(np.array([[-1], [0], [1]]))
    expected = np.array([[0.5], [0.73105858], [0.8807971]])

    assert np.all(np.isclose(actual, expected))


def test_network_forward_pass_multiple_layers():
    network = Network(
        layers=[
            Layer(
                input_neurons=1,
                neurons=3,
                activation_function="sigmoid",
                softmax=False,
                weights=np.array([[0.1, 0.2, -0.1]]),
                bias=True,
                bias_weights=np.array([[0.1, 0.1, 0.1]]),
            ),
            Layer(
                input_neurons=3,
                neurons=2,
                activation_function="sigmoid",
                softmax=False,
                weights=np.array([[0.1, -0.1], [-0.1, -0.1], [0.2, 0.1]]),
                bias=True,
                bias_weights=np.array([[0.2, 0.3]]),
            ),
        ],
        loss_function="mse",
        regularization=None,
    )

    data = np.array([[1], [0], [-1]])

    actual = network.forward_pass(data)
    expected = np.array(
        [[0.57384083, 0.55911531], [0.57566333, 0.56156158], [0.57748676, 0.56401704]]
    )

    assert np.all(np.isclose(actual, expected))


def test_network_with_input_layer():
    network = Network(
        layers=[
            InputLayer(),
            Layer(
                input_neurons=2,
                neurons=1,
                activation_function="sigmoid",
                softmax=False,
                weights=np.array([[0.1], [-0.3]]),
                bias=True,
                bias_weights=np.array([[0.2]]),
            ),
        ],
        loss_function="mse",
        regularization=None,
    )

    data = np.array([[1, 1], [0, 0], [-1, 1]])
    actual = network.forward_pass(data)
    expected = np.array([[0.5], [0.549834], [0.450166]])

    assert np.all(np.isclose(actual, expected))