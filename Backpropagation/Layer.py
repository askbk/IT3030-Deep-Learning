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
        initial_weight_range=(-0.1, 0.1),
        weights=None,
        bias=True,
        bias_weights=None,
    ):
        self._neurons = neurons
        if weights is not None:
            if weights.shape != (input_neurons, neurons):
                raise ValueError(
                    f"Weight matrix must have shape ({input_neurons}, {neurons}), was {weights.shape}"
                )
            self._weights = weights
        else:
            self._weights = np.random.uniform(
                low=initial_weight_range[0],
                high=initial_weight_range[1],
                size=(input_neurons, neurons),
            )

        if bias:
            if bias_weights is not None:
                if bias_weights.shape != (neurons,):
                    raise ValueError(
                        f"Bias matrix must have shape ({neurons}, ), was {bias_weights.shape}"
                    )
                self._bias = bias_weights
            else:
                self._bias = np.random.uniform(
                    low=initial_weight_range[0],
                    high=initial_weight_range[1],
                    size=neurons,
                )
        else:
            self._bias = np.zeros(shape=neurons)

        if activation_function not in ("sigmoid", "tanh", "relu", "linear"):
            raise ValueError("Invalid activation function.")

        self._activation_function = activation_function

    @staticmethod
    def _sigmoid(X):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _sigmoid_derivative(X):
        return Layer._sigmoid(X) * (1 - Layer._sigmoid(X))

    @staticmethod
    def _tanh(X):
        """
        Hyperbolic tangent
        """
        return np.tanh(X)

    @staticmethod
    def _tanh_derivative(X):
        """
        Hyperbolic tangent
        """
        return 1 - Layer._tanh(X) ** 2

    @staticmethod
    def _linear(X):
        """
        Linear function
        """
        return X

    @staticmethod
    def _linear_derivative(X):
        return np.ones_like(X)

    @staticmethod
    def _relu(X):
        """
        Rectified linear unit function
        """
        return np.maximum(X, 0)

    @staticmethod
    def _relu_derivative(X):
        return np.where(X <= 0, 0, 1)

    def _multiply_weights(self, X: np.array):
        return X @ self._weights

    def _add_bias(self, X: np.array):
        return X + self._bias

    def _apply_activation_function(self, data):
        """
        Applies the current activation function to the data.
        """
        if self._activation_function == "sigmoid":
            return Layer._sigmoid(data)

        if self._activation_function == "tanh":
            return Layer._tanh(data)

        if self._activation_function == "relu":
            return Layer._relu(data)

        if self._activation_function == "linear":
            return Layer._linear(data)

        raise NotImplementedError()

    def forward_pass(self, data):
        """
        Data is a row-vector representing a single test case.
        """
        return self._apply_activation_function(
            self._add_bias(self._multiply_weights(data))
        )
