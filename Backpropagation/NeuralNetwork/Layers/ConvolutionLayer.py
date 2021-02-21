import numpy as np
from scipy.signal import convolve
from NeuralNetwork.Layers import LayerBase


class ConvolutionLayer(LayerBase):
    def __init__(self, mode="valid", _kernel=None):
        self._mode = mode
        if _kernel is not None:
            self._kernel = _kernel

    def forward_pass(self, data):
        return convolve(self._kernel, data, mode=self._mode)