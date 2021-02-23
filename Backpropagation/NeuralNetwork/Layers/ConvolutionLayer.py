import math
import numpy as np
from itertools import product
from scipy.signal import convolve, correlate
from NeuralNetwork.Layers import LayerBase


class ConvolutionLayer(LayerBase):
    def __init__(self, mode="valid", stride=1, _kernels=None):
        if mode == "same" and stride > 1:
            raise ValueError(
                "Convolution mode 'same' and stride > 1 does not make sense."
            )
        self._mode = mode
        self._stride = stride
        if _kernels is not None:
            self._kernels = _kernels

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

    def forward_pass(self, data):
        channels, rows, columns = data.shape
        kernel_size = self._kernels.shape[-2:]
        out_channels = channels * len(self._kernels)
        output = np.empty(
            (out_channels,)
            + ConvolutionLayer._get_output_dimensions(
                self._mode, self._stride, kernel_size, (rows, columns)
            )
        )
        padding_x, padding_y = ConvolutionLayer._get_padding(
            self._mode, self._stride, kernel_size
        )
        max_row_index = rows + 2 * padding_x - kernel_size[0] + 1
        max_col_index = columns + 2 * padding_y - kernel_size[1] + 1
        for kernel_index, kernel in enumerate(self._kernels):
            for channel_index, channel in enumerate(data):
                output_channel_index = len(self._kernels) * kernel_index + channel_index
                padded_channel = np.pad(
                    channel, ((padding_x, padding_x), (padding_y, padding_y))
                )
                for out_row_index, input_row_index in enumerate(
                    range(0, max_row_index, self._stride)
                ):
                    for out_col_index, input_col_index in enumerate(
                        range(0, max_col_index, self._stride)
                    ):
                        padded_channel_slice = padded_channel[
                            input_row_index : input_row_index + kernel_size[0],
                            input_col_index : input_col_index + kernel_size[1],
                        ]
                        output[output_channel_index][out_row_index][
                            out_col_index
                        ] = np.einsum("ij, ij", kernel, padded_channel_slice)

        return output