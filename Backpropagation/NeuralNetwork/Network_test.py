import numpy as np
from Backpropagation.NeuralNetwork.Network import Network
from Backpropagation.NeuralNetwork.Layer import Layer
from Backpropagation.NeuralNetwork.InputLayer import InputLayer
from Backpropagation.NeuralNetwork.OutputLayer import OutputLayer
from Backpropagation.NeuralNetwork.Math import Loss
from Backpropagation.DataUtils import randomize_dataset


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
                activation_function="relu",
                use_bias=True,
            ),
            Layer(
                input_neurons=2,
                neurons=1,
                activation_function="relu",
                use_bias=True,
            ),
            OutputLayer(input_neurons=1),
        ],
        regularization=False,
        loss_function="mse",
        learning_rate=0.01,
    )

    minibatch_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    minibatch_y = [[0], [1], [1], [0]]
    minibatch_count = 5000
    X, Y = randomize_dataset(
        minibatch_x * minibatch_count, minibatch_y * minibatch_count
    )
    before_tranining = np.array([network.forward_pass(x) for x in minibatch_x])
    trained_network = network.train(
        np.array(X), np.array(Y), minibatches=minibatch_count
    )

    output_after_training = np.array(
        [trained_network.forward_pass(x) for x in minibatch_x]
    )
    print(output_after_training)
    print(
        f"Error before training: {Loss.mean_squared_error(before_tranining, minibatch_y)}\nError after training: {Loss.mean_squared_error(output_after_training, minibatch_y)}"
    )
