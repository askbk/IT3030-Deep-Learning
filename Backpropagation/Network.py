from functools import reduce
import numpy as np


class Network:
    """
    A neural network.
    """

    def __init__(self, layers, loss_function, regularization, regularization_rate=None):
        self._layers = layers
        self._loss_function = loss_function

    def forward_pass(self, X):
        """
        Forward pass through the network.
        """
        return reduce(lambda data, layer: layer.forward_pass(data), self._layers, X)

    @staticmethod
    def _mean_squared_error(Y: np.array, Y_hat: np.array):
        """
        Calculates mean squared error.
        """
        return ((Y - Y_hat) ** 2).mean(axis=1)

    def _apply_loss_function(self, Y, Y_hat):
        """
        docstring
        """
        if self._loss_function == "mse":
            return Network._mean_squared_error(Y, Y_hat)

        raise NotImplementedError

    def _train_minibatch(self, batch):
        X, Y = batch
        output = self.forward_pass(X)
        loss = self._apply_loss_function(Y, output)
        output_jacobian = self._layers[-1].backward_pass(loss)
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
