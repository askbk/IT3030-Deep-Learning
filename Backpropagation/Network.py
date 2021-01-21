from functools import reduce


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
        return reduce(lambda data, layer: layer.forward_pass(data), self._layers, X)

    def _train_minibatch(self, batch):
        return self

    def train(self, X, Y, minibatches=1):
        """
        Returns a new instance of the Network that is trained with X and Y.
        """
        minibatch_size = len(X) // minibatches
        X_batches = [
            X[i : i + minibatch_size] for i in range(0, len(X), minibatch_size)
        ]
        Y_batches = [
            Y[i : i + minibatch_size] for i in range(0, len(Y), minibatch_size)
        ]

        return reduce(
            lambda network, batch: network._train_minibatch(batch),
            zip(X_batches, Y_batches),
            self,
        )
