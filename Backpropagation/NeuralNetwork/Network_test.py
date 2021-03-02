import numpy as np
from NeuralNetwork import Network
from NeuralNetwork.Layers import DenseLayer, InputLayer, OutputLayer, ConvolutionLayer
from NeuralNetwork.Math import Loss
from DataUtils import randomize_dataset


def test_network_constructor():
    Network(
        layers=[], loss_function="mse", regularization=None, regularization_rate=0.001
    )


def test_network_forward_pass_base_case():
    network = Network(
        layers=[
            DenseLayer(
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
            DenseLayer(
                input_neurons=1,
                neurons=3,
                activation_function="sigmoid",
                weights=np.array([[0.1, 0.1, 0.1], [0.1, 0.2, -0.1]]),
                use_bias=True,
            ),
            DenseLayer(
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
            DenseLayer(
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


def test_network_with_conv_layer():
    kernel = np.array([[[1, 0], [-1, 1]]])
    weights = np.array(
        [
            [
                [[0.5, 0.9], [0.0, 0.7], [0.3, 0.9]],
                [
                    [0.2, 0.6],
                    [0.6, 0.2],
                    [0.3, 0.0],
                ],
                [
                    [0.6, 0.9],
                    [0.3, 0.5],
                    [0.7, 0.7],
                ],
            ]
        ]
    )
    assert weights.shape == (1, 3, 3, 2)
    network = Network(
        layers=[
            InputLayer(),
            ConvolutionLayer(_kernels=kernel, mode="valid"),
            DenseLayer(
                (1, 3, 3),
                neurons=2,
                activation_function="linear",
                weights=weights,
                use_bias=False,
            ),
        ],
        loss_function="mse",
        regularization=None,
    )

    data = np.array([[[1, 1, 0, 1], [3, 1, 2, 2], [0, 4, 5, 1], [3, 1, 2, 1]]])
    actual = network.forward_pass(data)
    _conv_result = np.array([[[-1, 2, 0], [7, 2, -2], [-2, 5, 4]]])
    expected = np.array([4.6, 8.6])

    assert np.all(np.isclose(actual, expected))

    network.train([(data, expected)], minibatches=1)


def test_multiple_conv_layers():
    network = Network(
        layers=[
            InputLayer(),
            ConvolutionLayer(mode="same", kernel_shape=(4, 4, 4), stride=1),
            ConvolutionLayer(mode="valid", kernel_shape=(4, 3, 3), stride=2),
            DenseLayer(
                (48, 7, 10),
                neurons=4,
                activation_function="linear",
                use_bias=False,
            ),
            OutputLayer(4, softmax=True),
        ],
        loss_function="cross_entropy",
        regularization=None,
    )

    assert network._layers[-2]._weights.shape == (48, 7, 10, 4)

    rng = np.random.default_rng()
    data = rng.random((3, 15, 21))
    expected = rng.random((4,))

    network.train([(data, expected)], minibatches=1)


def test_training_base_case():
    network = Network(
        layers=[
            InputLayer(),
            DenseLayer(
                input_neurons=2,
                neurons=2,
                activation_function="sigmoid",
                use_bias=True,
            ),
            DenseLayer(
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
            DenseLayer(
                input_neurons=2,
                neurons=2,
                activation_function="sigmoid",
                use_bias=True,
            ),
            DenseLayer(
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
