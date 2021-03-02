import math
import numpy as np
from more_itertools import intersperse
from itertools import product
from scipy.signal import convolve, correlate
from scipy.ndimage import grey_dilation
from NeuralNetwork.Layers import LayerBase


class ConvolutionLayer(LayerBase):
    def __init__(
        self,
        kernel_shape=None,
        mode="valid",
        stride=1,
        initial_weight_range=(-0.1, 0.1),
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
        if _kernels is not None:
            self._kernels = _kernels

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
    def _get_output_dimensions(mode, stride, kernel_size, data_size):
        if mode == "same":
            return data_size

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

        # interspersed_elements = np.array(
        #     [intersperse(0, row, n=dilation_factor) for row in array]
        # )
        # return np.array(
        #     intersperse(
        #         np.zeros(interspersed_elements.shape[-1]),
        #         interspersed_elements,
        #         n=dilation_factor,
        #     )
        # )

    @staticmethod
    def _pad_3d_array(array, padding_x, padding_y):
        return np.pad(array, ((0, 0), (padding_x, padding_x), (padding_y, padding_y)))

    @staticmethod
    def _pad_2d_array(array, padding_x, padding_y):
        return np.pad(array, ((padding_x, padding_x), (padding_y, padding_y)))

    @staticmethod
    def _correlate(data, kernels, mode="valid", stride=1):
        channels, rows, columns = data.shape
        kernel_size = kernels.shape[-2:]
        out_channels = channels * len(kernels)
        output = np.empty(
            (out_channels,)
            + ConvolutionLayer._get_output_dimensions(
                mode, stride, kernel_size, (rows, columns)
            )
        )
        padding_x, padding_y = ConvolutionLayer._get_padding(mode, stride, kernel_size)
        max_row_index = ConvolutionLayer._get_max_input_index(
            mode, kernel_size[0], rows, padding_x
        )
        max_col_index = ConvolutionLayer._get_max_input_index(
            mode, kernel_size[1], columns, padding_y
        )
        for kernel_index, kernel in enumerate(kernels):
            for channel_index, channel in enumerate(data):
                output_channel_index = channels * kernel_index + channel_index
                padded_channel = ConvolutionLayer._pad_2d_array(
                    channel, padding_x, padding_y
                )
                for out_row_index, input_row_index in enumerate(
                    range(0, max_row_index, stride)
                ):
                    for out_col_index, input_col_index in enumerate(
                        range(0, max_col_index, stride)
                    ):
                        padded_channel_slice = padded_channel[
                            input_row_index : input_row_index + kernel_size[0],
                            input_col_index : input_col_index + kernel_size[1],
                        ]
                        output[output_channel_index][out_row_index][
                            out_col_index
                        ] = np.einsum("ij, ij", kernel, padded_channel_slice)
        return output

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
        return np.array(
            [
                np.sum(array[start : start + channel_interval], axis=0)
                for start in range(target_channel_count)
            ]
        )

    def backward_pass(self, J_L_Y, Y, X):
        padding_x, padding_y = self._kernels.shape[-2] - 1, self._kernels.shape[-1] - 1
        dilated_output = ConvolutionLayer._dilate_output(
            Y,
            dilation_factor=self._stride - 1,
        )

        padded_dilated_output = ConvolutionLayer._pad_3d_array(
            dilated_output, padding_x, padding_y
        )
        print(dilated_output.shape, X.shape)

        J_L_W = ConvolutionLayer._sum_over_channel_intervals(
            ConvolutionLayer._correlate(X, dilated_output, mode=self._mode),
            self._kernels.shape[0],
        )
        flipped_kernel = np.fliplr(np.flipud(self._kernels))
        J_L_X = ConvolutionLayer._sum_over_channel_intervals(
            ConvolutionLayer._correlate(
                padded_dilated_output, flipped_kernel, mode=self._mode
            ),
            X.shape[0],
        )
        print(J_L_W.shape, self._kernels.shape, J_L_W.shape == self._kernels.shape)
        assert J_L_W.shape == self._kernels.shape
        # print(J_L_X.shape, padded_dilated_output.shape, flipped_kernel.shape)
        # print("goal shape:", X.shape)
        assert J_L_X.shape == X.shape

        return J_L_W, J_L_X

    def forward_pass(self, data):
        return ConvolutionLayer._correlate(
            data, self._kernels, mode=self._mode, stride=self._stride
        )
