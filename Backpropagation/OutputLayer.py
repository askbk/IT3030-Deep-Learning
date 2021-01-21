import numpy as np


class OutputLayer:
    """
    An output layer.
    """

    def __init__(self, softmax=False):
        self._use_softmax = softmax

    @staticmethod
    def _softmax(X):
        """
        The softmax function.
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def forward_pass(self, data):
        if self._use_softmax:
            return OutputLayer._softmax(data)

        return data