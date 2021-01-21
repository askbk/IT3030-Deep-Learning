class Network:
    """
    A neural network.
    """

    def __init__(self, layers, loss_function, regularization, regularization_rate=None):
        self._layers = layers

    def forward_pass(self, X):
        """
        Forward pass through the network.
        """
        return self._layers[0].forward_pass(X)