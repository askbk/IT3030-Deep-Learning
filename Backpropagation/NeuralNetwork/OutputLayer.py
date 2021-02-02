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
        return np.exp(X) / np.sum(np.exp(X), axis=0, keepdims=True)

    @staticmethod
    def _softmax_derivative(X):
        X_reshape = X.reshape((-1, 1))
        return np.diagflat(X) - np.dot(X_reshape, X_reshape.T)

    def backward_pass(self, J_L_S, S, Z):
        """
        Computes Jacobian.
        """
        if not self._use_softmax:
            return None, J_L_S

        J_S_Z = OutputLayer._softmax_derivative(Z)
        J_L_Z = np.dot(J_L_S, J_S_Z)

        return None, J_L_Z

    def forward_pass(self, data):
        if not self._use_softmax:
            return data

        return OutputLayer._softmax(data)

    def update_weights(self, jacobians, learning_rate):
        return self