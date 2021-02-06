import numpy as np
from NeuralNetwork.Network import Network
from NeuralNetwork.Layer import Layer
from NeuralNetwork.InputLayer import InputLayer
from NeuralNetwork.OutputLayer import OutputLayer
from NeuralNetwork.Math import Loss
from DataUtils import randomize_dataset


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
                weights=np.array([[1], [1]]),
                use_bias=True,
            )
        ],
        loss_function="mse",
        regularization=None,
    )

    cases = np.array([[-1], [0], [1]])
    actual = [network.forward_pass(case) for case in cases]
    expected = np.array([[0.5], [0.73105858], [0.8807971]])

    assert np.all(np.isclose(actual, expected))


def test_network_forward_pass_multiple_layers():
    network = Network(
        layers=[
            Layer(
                input_neurons=1,
                neurons=3,
                activation_function="sigmoid",
                weights=np.array([[0.1, 0.1, 0.1], [0.1, 0.2, -0.1]]),
                use_bias=True,
            ),
            Layer(
                input_neurons=3,
                neurons=2,
                activation_function="sigmoid",
                weights=np.array([[0.2, 0.3], [0.1, -0.1], [-0.1, -0.1], [0.2, 0.1]]),
                use_bias=True,
            ),
        ],
        loss_function="mse",
        regularization=None,
    )

    cases = np.array([[1], [0], [-1]])

    actual = [network.forward_pass(case) for case in cases]
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
                weights=np.array([[0.2], [0.1], [-0.3]]),
                use_bias=True,
            ),
        ],
        loss_function="mse",
        regularization=None,
    )

    cases = np.array([[1, 1], [0, 0], [-1, 1]])
    actual = [network.forward_pass(case) for case in cases]
    expected = np.array([[0.5], [0.549834], [0.450166]])

    assert np.all(np.isclose(actual, expected))


def test_training_base_case():
    network = Network(
        layers=[
            InputLayer(),
            Layer(
                input_neurons=2,
                neurons=2,
                activation_function="sigmoid",
                use_bias=True,
            ),
            Layer(
                input_neurons=2,
                neurons=1,
                activation_function="sigmoid",
                use_bias=True,
            ),
            OutputLayer(input_neurons=1),
        ],
        regularization="l2",
        regularization_rate=0.001,
        loss_function="mse",
        learning_rate=0.01,
    )

    minibatch = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]
    minibatch_count = 10
    dataset = randomize_dataset(minibatch * minibatch_count)
    _trained_network, *_ = network.train(
        dataset, minibatches=minibatch_count, verbose=True, validation_set=minibatch
    )


def test_network_test():
    network = Network(
        layers=[
            InputLayer(),
            Layer(
                input_neurons=2,
                neurons=2,
                activation_function="sigmoid",
                use_bias=True,
            ),
            Layer(
                input_neurons=2,
                neurons=1,
                activation_function="sigmoid",
                use_bias=True,
            ),
            OutputLayer(input_neurons=1),
        ],
        regularization="l2",
        regularization_rate=0.001,
        loss_function="mse",
        learning_rate=0.01,
    )

    minibatch = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]
    minibatch_count = 500
    dataset = randomize_dataset(minibatch * minibatch_count)
    trained_network, *_ = network.train(dataset, minibatches=minibatch_count)

    test_performance = trained_network.test(minibatch)

    assert test_performance is not None