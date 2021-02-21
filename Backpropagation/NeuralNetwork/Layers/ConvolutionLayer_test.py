import numpy as np
from NeuralNetwork.Layers import ConvolutionLayer


def test_conv_layer_constructor():
    ConvolutionLayer()


def test_1d_forward_pass():
    kernel = np.array([1, 2, -1])
    data = np.array([1, 4, 5, 1, 3, 1, 2, 2])
    layer = ConvolutionLayer(_kernel=kernel, mode="valid")
    correct_output = np.array([4, 13, 4, 6, 3, 3])
    output = layer.forward_pass(data)

    assert np.all(np.isclose(output, correct_output))


def test_2d_forward_pass():
    kernel = np.array([[1, 0], [-1, 1]])
    data = np.array([[1, 1, 0, 1], [3, 1, 2, 2], [0, 4, 5, 1], [3, 1, 2, 1]])
    layer = ConvolutionLayer(_kernel=kernel, mode="valid")
    correct_output = np.array([[-1, 2, 0], [7, 2, -2], [-2, 5, 4]])
    output = layer.forward_pass(data)

    assert np.all(np.isclose(output, correct_output))