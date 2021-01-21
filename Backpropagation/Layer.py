import numpy as np


class Layer:
    """
    A layer in a neural network.
    """

    def __init__(
        self,
        input_neurons,
        neurons,
        activation_function="sigmoid",
        softmax=False,
        initial_weight_range=(-0.1, 0.1),
        weights=None,
        bias=True,
        bias_weights=None,
    ):
        self._neurons = neurons
        if weights is not None:
            if weights.shape != (input_neurons, neurons):
                raise ValueError(
                    f"weight matrix must have shape ({input_neurons}, {neurons}), was {weights.shape}"
                )
            self._weights = weights
        else:
            self._weights = np.random.uniform(
                low=initial_weight_range[0],
                high=initial_weight_range[1],
                size=(input_neurons, neurons),
            )

        if bias:
            if bias_weights:
                self._bias = bias_weights
            else:
                self._bias = np.random.uniform(
                    low=initial_weight_range[0],
                    high=initial_weight_range[1],
                    size=neurons,
                )
        else:
            self._bias = np.zeros(neurons)

    @staticmethod
    def _sigmoid(X):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-X))

    def _multiply_weights_input(self, X: np.array):
        return X @ self._weights

    def _add_bias(self, X: np.array):
        bias = np.broadcast_to(self._bias, (X.shape[0], 1))
        return X + bias

    def forward_pass(self, data):
        """
        Data is a matrix with a minibatch of data.
        Each test case should be row-oriented.
        """
        product = self._multiply_weights_input(data)
        with_bias = self._add_bias(product)
        return Layer._sigmoid(np.sum(with_bias, axis=1, keepdims=True))