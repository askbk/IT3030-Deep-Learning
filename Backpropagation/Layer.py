import numpy as np


class Layer:
    """
    A layer in a neural network.
    """

    def __init__(
        self,
        input_neurons=5,
        neurons=5,
        activation_function="sigmoid",
        softmax=False,
        initial_weight_range=(-0.1, 0.1),
        weights=None,
    ):
        self._neurons = neurons
        if weights is not None:
            self._weights = weights
        else:
            self._weights = np.random.uniform(
                low=initial_weight_range[0],
                high=initial_weight_range[1],
                size=(neurons, input_neurons),
            )

    @staticmethod
    def _sigmoid(X):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-X))

    def _multiply_weights_input(self, X: np.array):
        """
        docstring
        """
        return X @ self._weights

    def forward_pass(self, data):
        """
        Data is a minibatch of data.
        """
        return Layer._sigmoid(np.sum(self._multiply_weights_input(data)))