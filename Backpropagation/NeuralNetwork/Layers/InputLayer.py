import numpy as np
from NeuralNetwork.Layers import LayerBase


class InputLayer(LayerBase):
    """
    Represents an input layer with no weights or biases.
    """

    def forward_pass(self, data):
        """
        Passes the data through the layer without modifying it.
        """
        return data

    def backward_pass(self, J_L_Z, Z, Y):
        """
        Returns a tuple of the weight Jacobian and the Jacobian to pass upstream.
        """
        return None, None

    def update_weights(self, jacobians, learning_rate):
        return self

    def get_weights(self):
        return None