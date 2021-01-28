import numpy as np


class OutputLayer:
    """
    An output layer.
    """

    def __init__(self, input_neurons, softmax=False):
        self._use_softmax = softmax
        self._input_neurons = input_neurons

    @staticmethod
    def _softmax(X):
        """
        The softmax function.
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def backward_pass(self, J_L_Z, Z, Y):
        """
        Computes Jacobian.
        """
        if self._use_softmax:
            raise NotImplementedError

        return None, J_L_Z

    def forward_pass(self, data):
        if self._use_softmax:
            return OutputLayer._softmax(data)

        return data

    def update_weights(self, jacobians, learning_rate):
        if self._use_softmax:
            raise NotImplementedError

        return self