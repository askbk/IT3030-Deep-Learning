import numpy as np
from itertools import product
from scipy.signal import convolve, correlate
from NeuralNetwork.Layers import LayerBase


class ConvolutionLayer(LayerBase):
    def __init__(self, mode="valid", _kernels=None):
        self._mode = mode
        if _kernels is not None:
            self._kernels = _kernels

    def forward_pass(self, data):
        return np.array(
            [
                correlate(channel, kernel, mode=self._mode)
                for kernel, channel in product(self._kernels, data)
            ]
        )
