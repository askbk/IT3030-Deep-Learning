import numpy as np
from NeuralNetwork.Math import Activation
from NeuralNetwork.Layers import LayerBase


class OutputLayer(LayerBase):
    """
    An output layer.
    """

    def __init__(self, input_neurons, softmax=False):
        self._use_softmax = softmax
        self._input_neurons = input_neurons

    def backward_pass(self, J_L_S, S, X):
        """
        Computes Jacobian.
        """
        if not self._use_softmax:
            return None, J_L_S

        J_S_X = Activation.softmax_derivative(X)
        J_L_X = np.dot(J_L_S, J_S_X)

        return None, J_L_X

    def forward_pass(self, data):
        if not self._use_softmax:
            return data

        return Activation.softmax(data)

    def update_weights(self, jacobians, learning_rate):
        return self

    def get_weights(self):
        return None
