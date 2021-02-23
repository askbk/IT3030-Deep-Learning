import numpy as np
from NeuralNetwork.Layers import ConvolutionLayer


def test_conv_layer_constructor():
    ConvolutionLayer()


def test_1d_forward_pass():
    kernel = np.array([[1, 2, -1]])
    data = np.array([[1, 4, 5, 1, 3, 1, 2, 2]])
    layer = ConvolutionLayer(_kernels=kernel, mode="valid")
    correct_output = np.array([[4, 13, 4, 6, 3, 3]])
    output = layer.forward_pass(data)

    assert np.all(np.isclose(output, correct_output))


def test_2d_forward_pass():
    kernel = np.array([[[1, 0], [-1, 1]]])
    data = np.array([[[1, 1, 0, 1], [3, 1, 2, 2], [0, 4, 5, 1], [3, 1, 2, 1]]])
    layer = ConvolutionLayer(_kernels=kernel, mode="valid")
    correct_output = np.array([[[-1, 2, 0], [7, 2, -2], [-2, 5, 4]]])
    output = layer.forward_pass(data)

    assert np.all(np.isclose(output, correct_output))


def test_multichannel_2d_forward_pass():
    kernels = np.array([[[1, 0], [-1, 1]], [[0, 1], [1, 0]]])
    data = np.array(
        [
            [[1, 1, 0, 1], [3, 1, 2, 2], [0, 4, 5, 1], [3, 1, 2, 1]],
            [[0, 1, 0, 1], [1, 0, 3, 2], [4, 0, 1, 5], [1, 2, 0, 0]],
        ]
    )
    layer = ConvolutionLayer(_kernels=kernels, mode="valid")
    correct_output = np.array(
        [
            [[-1, 2, 0], [7, 2, -2], [-2, 5, 4]],
            [[-1, 4, -1], [-3, 1, 7], [5, -2, 1]],
            [[4, 1, 3], [1, 6, 7], [7, 6, 3]],
            [[2, 0, 4], [4, 3, 3], [1, 3, 5]],
        ]
    )
    output = layer.forward_pass(data)
    assert np.all(np.isclose(output, correct_output))