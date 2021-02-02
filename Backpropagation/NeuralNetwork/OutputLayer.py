import numpy as np
from NeuralNetwork.Math import Activation


class OutputLayer:
    """
    An output layer.
    """

    def __init__(self, input_neurons, softmax=False):
        self._use_softmax = softmax
        self._input_neurons = input_neurons

    def backward_pass(self, J_L_S, S, Z):
        """
        Computes Jacobian.
        """
        if not self._use_softmax:
            return None, J_L_S

        J_S_Z = Activation.softmax_derivative(Z)
        J_L_Z = np.dot(J_L_S, J_S_Z)

        return None, J_L_Z

    def forward_pass(self, data):
        if not self._use_softmax:
            return data

        return Activation.softmax(data)

    def update_weights(self, jacobians, learning_rate):
        return self