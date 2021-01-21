from Network import Network
from Layer import Layer
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
                weights=[[1]],
                bias=True,
                bias_weights=[1],
            )
        ],
        loss_function="mse",
        regularization=None,
    )

    actual = network.forward_pass(np.array([[-1], [0], [1]]))
    expected = np.array([[0.5], [0.73105858], [0.8807971]])

    assert np.all(np.isclose(actual, expected))