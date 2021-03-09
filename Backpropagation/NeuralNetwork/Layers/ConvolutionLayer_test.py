import numpy as np
from itertools import product
from NeuralNetwork.Layers import ConvolutionLayer


def test_conv_layer_constructor():
    ConvolutionLayer(kernel_shape=(1, 1, 2))


def test_sigmoid_activation():
    kernel = np.array([[[1, 2, -1]]])
    data = np.array([[[1, 4, 5, 1, 3, 1, 2, 2]]])
    layer2 = ConvolutionLayer(
        _kernels=kernel, mode="same", activation_function="sigmoid"
    )
    correct_output2 = np.array(
        [
            [
                [
                    0.11920292,
                    0.98201379,
                    0.99999774,
                    0.98201379,
                    0.99752738,
                    0.95257413,
                    0.95257413,
                    0.99752738,
                ]
            ]
        ]
    )

    output2 = layer2.forward_pass(data)

    assert np.all(np.isclose(output2, correct_output2))


def test_1d_forward_pass():
    kernel = np.array([[[1, 2, -1]]])
    data = np.array([[[1, 4, 5, 1, 3, 1, 2, 2]]])

    layer1 = ConvolutionLayer(_kernels=kernel, mode="valid")
    correct_output1 = np.array([[[4, 13, 4, 6, 3, 3]]])
    output1 = layer1.forward_pass(data)

    assert np.all(np.isclose(output1, correct_output1))

    layer2 = ConvolutionLayer(_kernels=kernel, mode="same")
    correct_output2 = np.array([[[-2, 4, 13, 4, 6, 3, 3, 6]]])
    output2 = layer2.forward_pass(data)

    assert np.all(np.isclose(output2, correct_output2))

    layer3 = ConvolutionLayer(_kernels=kernel, mode="full")
    correct_output3 = np.array([[[-1, -2, 4, 13, 4, 6, 3, 3, 6, 2]]])
    output3 = layer3.forward_pass(data)

    assert np.all(np.isclose(output3, correct_output3))


def test_1d_forward_pass_stride_2():
    kernel = np.array([[[1, 0, -1]]])
    data = np.array([[[1, 4, 0, 5, 6, 7]]])
    expected_valid = np.array([[[1, -6]]])
    expected_full = np.array([[[-1, 1, -6, 6]]])
    output_valid = ConvolutionLayer(
        _kernels=kernel, mode="valid", stride=2
    ).forward_pass(data)
    output_full = ConvolutionLayer(_kernels=kernel, mode="full", stride=2).forward_pass(
        data
    )

    assert np.all(np.isclose(output_valid, expected_valid))
    assert np.all(np.isclose(output_full, expected_full))


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
    layer = ConvolutionLayer(_kernels=kernels, mode="valid", activation_function="relu")
    correct_output = np.array(
        [
            [[0, 2, 0], [7, 2, 0], [0, 5, 4]],
            [[0, 4, 0], [0, 1, 7], [5, 0, 1]],
            [[4, 1, 3], [1, 6, 7], [7, 6, 3]],
            [[2, 0, 4], [4, 3, 3], [1, 3, 5]],
        ]
    )
    output = layer.forward_pass(data)

    assert np.all(np.isclose(output, correct_output))


def test_1d_backward_pass():
    kernels = np.array([[[1, 1, -1]]])
    data = np.array([[[0, 0, 1, 1, 1, 1, 0, 0, 0]]])
    layer = ConvolutionLayer(_kernels=kernels, mode="valid", stride=1)
    forward_output = layer.forward_pass(data)
    assert np.all(np.isclose(forward_output, np.array([-1, 0, 1, 1, 2, 1, 0])))

    J_L_Z = np.array([[[0.1, 0, 0.1, 0.1, -0.2, 0.1, 0]]])
    _backward_output = layer.backward_pass(J_L_Z, forward_output, data)


# def test_1d_backward_pass_full_mode():
#     # kernels = np.array([[[1, 1, -1]]])
#     # X = np.array([[[0, 0, 1, 1, 1, 1, 0, 0, 0]]])

#     # layer1 = ConvolutionLayer(_kernels=kernels, mode="full", stride=1)
#     # Y1 = layer1.forward_pass(X)
#     # J_L_Y1 = np.array([[[0.1, 0, 0.1, 0.1, -0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.1]]])
#     # layer1.backward_pass(J_L_Y1, Y1, X)

#     for kernel_size, X_size, stride in product(range(1, 6), range(4, 10), range(1, 4)):
#         kernels = np.arange(kernel_size).reshape((1, 1, kernel_size))
#         X = np.arange(X_size).reshape((1, 1, X_size))
#         layer = ConvolutionLayer(_kernels=kernels, mode="full", stride=stride)
#         Y = layer.forward_pass(X)
#         JLY = np.ones_like(Y)
#         layer.backward_pass(JLY, Y, X)


# def test_1d_backward_pass_same_mode():
#     for kernel_size, X_size in product(range(1, 6), range(4, 10)):
#         kernels = np.arange(kernel_size).reshape((1, 1, kernel_size))
#         X = np.arange(X_size).reshape((1, 1, X_size))
#         layer = ConvolutionLayer(_kernels=kernels, mode="same", stride=1)
#         Y = layer.forward_pass(X)
#         JLY = np.ones_like(Y)
#         layer.backward_pass(JLY, Y, X)


def test_backward_pass_channel_ordering():
    kernels = np.array([[[1, 1]], [[0, 0]]])
    X = np.array([[[0, 0, 0]], [[1, 1, 1]]])
    Y = np.array([[[0, 0]], [[2, 2]], [[0, 0]], [[0, 0]]])
    JLY = np.array([[[0, 0]], [[1, 1]], [[0.5, 0.5]], [[0, 0]]])
    JLX_expected = np.array([[[0, 0, 0]], [[0.5, 1, 0.5]]])
    JLW_expected = np.array([[[2, 2]], [[0, 0]]])
    layer = ConvolutionLayer(_kernels=kernels, mode="valid", stride=1)
    JLW_actual, JLX_actual = layer.backward_pass(JLY, Y, X)
    assert np.all(np.isclose(layer.forward_pass(X), Y))
    assert np.all(np.isclose(JLW_actual, JLW_expected))
    assert np.all(np.isclose(JLX_actual, JLX_expected))


def test_2d_backward_pass():
    kernels = np.array([[[1, 1, -1], [0, 1, 0]]])
    data = np.array(
        [
            [
                [0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0],
            ]
        ]
    )
    layer = ConvolutionLayer(_kernels=kernels, mode="valid", stride=1)
    forward_output = layer.forward_pass(data)
    J_L_Y = np.array(
        [
            [
                [0.1, 0, 0.1, 0.1, -0.2, 0.1, 0],
                [0.1, 0, 0.1, 0.1, -0.2, 0.1, 0],
            ]
        ]
    )
    _backward_output = layer.backward_pass(J_L_Y, forward_output, data)
