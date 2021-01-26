import numpy as np


class InputLayer:
    """
    Represents an input layer with no weights or biases.
    """

    def forward_pass(self, data):
        """
        Passes the data through the layer without modifying it.
        """
        return data