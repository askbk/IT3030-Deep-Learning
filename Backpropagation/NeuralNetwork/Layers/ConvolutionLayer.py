import math
import numpy as np
from more_itertools import intersperse, distribute, divide
from itertools import product
from scipy.signal import convolve, correlate
from scipy.ndimage import grey_dilation
from NeuralNetwork.Layers import LayerBase
from NeuralNetwork.Math import Activation


class ConvolutionLayer(LayerBase):
    def __init__(
        self,
        kernel_shape=None,
        mode="valid",
        stride=1,
        initial_weight_range=(-0.1, 0.1),
        activation_function="linear",
        input_neurons=None,
        _kernels=None,
    ):
        if mode == "same" and stride > 1:
            raise ValueError(
                "Convolution mode 'same' and stride > 1 does not make sense."
            )
        self._mode = mode
        self._stride = stride
        self._kernels = ConvolutionLayer._initialize_kernels(
            kernel_shape, _kernels, initial_weight_range
        )
        self._activation_function = activation_function

        self._neurons = (
            None
            if input_neurons is None
            else ConvolutionLayer._get_output_neurons(
                mode, stride, self._kernels.shape, input_neurons
            )
        )

    @staticmethod
    def _initialize_kernels(kernel_shape, kernels, initial_weight_range):
        if kernels is not None:
            return kernels

        return np.random.default_rng().uniform(
            low=initial_weight_range[0], high=initial_weight_range[1], size=kernel_shape
        )

    @staticmethod
    def _get_padding(mode, stride, kernel_size):
        kernel_x, kernel_y = kernel_size
        if mode == "full":
            padding_x = kernel_x - 1
            padding_y = kernel_y - 1
            return padding_x, padding_y

        if mode == "valid":
            return 0, 0

        if mode == "same":
            return kernel_x // 2, kernel_y // 2

        raise ValueError("Mode must be either 'full', 'valid' or 'same'.")

    @staticmethod
    def _get_output_neurons(mode, stride, kernel_shape, data_shape):
        channels = kernel_shape[0] * data_shape[0]
        return (channels,) + ConvolutionLayer._get_output_dimensions(
            mode, stride, kernel_shape[-2:], data_shape[-2:]
        )

    @staticmethod
    def _get_output_dimensions(mode, stride, kernel_size, data_size):
        if mode == "same":
            return data_size
        # print(kernel_size, data_size)
        kernel_x, kernel_y = kernel_size
        data_x, data_y = data_size

        if mode == "valid":
            return math.ceil((data_x - kernel_x + 1) / stride), math.ceil(
                (data_y - kernel_y + 1) / stride
            )

        if mode == "full":
            return math.ceil((data_x + kernel_x - 1) / stride), math.ceil(
                (data_y + kernel_y - 1) / stride
            )

        raise ValueError("Mode must be either 'same', 'valid' or 'full'.")

    @staticmethod
    def _get_max_input_index(mode, kernel_size, data_size, padding):
        if mode == "full":
            return data_size + 2 * padding - kernel_size + 1

        if mode == "same":
            return data_size + 2 * padding - kernel_size + 1

        return data_size + padding - kernel_size + 1

    @staticmethod
    def _dilate_output(array, dilation_factor):
        if dilation_factor == 0:
            return array

        def dilate_channel(channel):
            interspersed_elements = np.array(
                [list(intersperse(0, row, n=dilation_factor)) for row in channel]
            )

            return list(
                intersperse(
                    np.zeros(interspersed_elements.shape[-1]),
                    interspersed_elements,
                    n=dilation_factor,
                )
            )

        return np.array([dilate_channel(channel) for channel in array])

    @staticmethod
    def _pad_3d_array(array, padding_x, padding_y):
        return np.pad(array, ((0, 0), (padding_x, padding_x), (padding_y, padding_y)))

    @staticmethod
    def _pad_2d_array(array, padding_x, padding_y):
        return np.pad(array, ((padding_x, padding_x), (padding_y, padding_y)))

    @staticmethod
    def _single_kernel_channel_correlation(
        kernel, channel, max_row_index, max_col_index, stride
    ):
        return np.array(
            [
                [
                    np.sum(
                        kernel
                        * channel[
                            input_row_index : input_row_index + kernel.shape[0],
                            input_col_index : input_col_index + kernel.shape[1],
                        ]
                    )
                    for input_col_index in range(0, max_col_index, stride)
                ]
                for input_row_index in range(0, max_row_index, stride)
            ]
        )

    @staticmethod
    def _channel_generator(data, target_channels, reorder):
        func = distribute if reorder else divide
        for group in func(target_channels, data):
            yield group

    @staticmethod
    def _correlate(data, kernels, mode="valid", stride=1):
        _channels, rows, columns = data.shape
        kernel_size = kernels.shape[-2:]
        padding_x, padding_y = ConvolutionLayer._get_padding(mode, stride, kernel_size)
        max_row_index = ConvolutionLayer._get_max_input_index(
            mode, kernel_size[0], rows, padding_x
        )
        max_col_index = ConvolutionLayer._get_max_input_index(
            mode, kernel_size[1], columns, padding_y
        )

        data_with_padding = ConvolutionLayer._pad_3d_array(data, padding_x, padding_y)

        return np.array(
            [
                ConvolutionLayer._single_kernel_channel_correlation(
                    kernel, channel, max_row_index, max_col_index, stride
                )
                for kernel in kernels
                for channel in data_with_padding
            ]
        )

    @staticmethod
    def _backward_correlate(in1, in2, mode, target_channels, reorder, stride=1):
        kernels = in1 if in1.shape[-2:] < in2.shape[-2:] else in2
        data = in2 if in1.shape[-2:] < in2.shape[-2:] else in1
        _channels, rows, columns = data.shape
        kernel_size = kernels.shape[-2:]
        padding_x, padding_y = ConvolutionLayer._get_padding(mode, stride, kernel_size)
        max_row_index = ConvolutionLayer._get_max_input_index(
            mode, kernel_size[0], rows, padding_x
        )
        max_col_index = ConvolutionLayer._get_max_input_index(
            mode, kernel_size[1], columns, padding_y
        )

        data_with_padding = ConvolutionLayer._pad_3d_array(data, padding_x, padding_y)
        return np.array(
            [
                np.sum(
                    [
                        ConvolutionLayer._single_kernel_channel_correlation(
                            kernel, channel, max_row_index, max_col_index, stride
                        )
                        for channel in channel_group
                    ],
                    axis=0,
                )
                for kernel, channel_group in zip(
                    kernels,
                    ConvolutionLayer._channel_generator(
                        data_with_padding, target_channels, reorder
                    ),
                )
            ]
        )

    def _apply_activation_function(self, S):
        if self._activation_function == "relu":
            return Activation.relu(S)

        if self._activation_function == "linear":
            return Activation.linear(S)

        if self._activation_function == "sigmoid":
            return Activation.sigmoid(S)

        if self._activation_function == "tanh":
            return Activation.tanh(S)

        raise NotImplementedError(f"{self._activation_function} not implemented")

    def _apply_activation_function_derivative(self, Y):
        if self._activation_function == "relu":
            return Activation.relu_derivative(Y)

        if self._activation_function == "linear":
            return Activation.linear_derivative(Y)

        if self._activation_function == "sigmoid":
            return Activation.sigmoid_derivative(Y)

        if self._activation_function == "tanh":
            return Activation.tanh_derivative(Y)

        raise NotImplementedError(f"{self._activation_function} not implemented")

    def get_weights(self):
        return self._kernels

    def update_weights(self, jacobians, learning_rate):
        return ConvolutionLayer(
            mode=self._mode,
            stride=self._stride,
            _kernels=self._kernels - learning_rate * np.sum(jacobians, axis=0),
        )

    @staticmethod
    def _sum_over_channel_intervals(array, target_channel_count):
        channel_count = array.shape[0]
        channel_interval = channel_count // target_channel_count
        if len(array.shape) != 3:
            print(array.shape, array)
            raise ValueError("array must be 3d")
        return np.array(
            [
                np.sum(array[start : start + channel_interval], axis=0)
                for start in range(target_channel_count)
            ]
        )

    def _calculate_JLW(self, dilated_JLY, X):
        if self._mode == "full":
            return ConvolutionLayer._sum_over_channel_intervals(
                ConvolutionLayer._backward_correlate(
                    X,
                    dilated_JLY,
                    mode="valid",
                    reorder=False,
                    target_channels=self._kernels.shape[0],
                ),
                self._kernels.shape[0],
            )

        if self._mode == "same":
            return ConvolutionLayer._sum_over_channel_intervals(
                ConvolutionLayer._backward_correlate(
                    X,
                    dilated_JLY,
                    mode="valid",
                    target_channels=self._kernels.shape[0],
                    reorder=False,
                ),
                self._kernels.shape[0],
            )

        if self._mode == "valid":
            return ConvolutionLayer._sum_over_channel_intervals(
                ConvolutionLayer._backward_correlate(
                    X,
                    dilated_JLY,
                    mode="valid",
                    target_channels=self._kernels.shape[0],
                    reorder=False,
                ),
                self._kernels.shape[0],
            )

        raise NotImplementedError

    def _calculate_JLX(self, dilated_JLY, X):
        return ConvolutionLayer._backward_correlate(
            dilated_JLY,
            np.fliplr(np.flipud(self._kernels)),
            mode="full",
            stride=1,
            reorder=True,
            target_channels=X.shape[0],
        )

    @staticmethod
    def _convert_to_3d(data):
        if len(data.shape) == 1:
            return data.reshape((1, 1, len(data)))
        if len(data.shape) == 2:
            return data.reshape((1, *data.shape))
        if len(data.shape) == 3:
            return data

    def backward_pass(self, JLY, Y, X):
        # use JLS = JLY * df(Y) instead of JLY when implementing activation functions
        JLS = JLY * self._apply_activation_function_derivative(Y)
        dilated_JLS = ConvolutionLayer._dilate_output(
            JLS,
            dilation_factor=self._stride - 1,
        )
        reshaped_X = ConvolutionLayer._convert_to_3d(X)
        JLX = self._calculate_JLX(dilated_JLS, reshaped_X)
        JLW = self._calculate_JLW(dilated_JLS, reshaped_X)

        if JLW.shape != self._kernels.shape:
            raise Exception(
                f"JLW shape: {JLW.shape}, kernel shape: {self._kernels.shape}, mode: {self._mode}"
            )
        if JLX.shape != reshaped_X.shape:
            raise Exception(
                f"JLX shape: {JLX.shape}, X shape: {reshaped_X.shape}, mode: {self._mode}"
            )

        return JLW, JLX

    def forward_pass(self, data):
        if len(data.shape) == 1:
            reshaped = data.reshape((1, 1, len(data)))
        if len(data.shape) == 2:
            reshaped = data.reshape((1, *data.shape))
        if len(data.shape) == 3:
            reshaped = data

        return self._apply_activation_function(
            ConvolutionLayer._correlate(
                reshaped, self._kernels, mode=self._mode, stride=self._stride
            )
        )

    def __repr__(self):
        return f"ConvolutionLayer<kernel_shape={self._kernels.shape}, mode={self._mode}, stride={self._stride}>"
