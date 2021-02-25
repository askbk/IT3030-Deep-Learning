import numpy as np
from NeuralNetwork.Math import Activation
from NeuralNetwork.Layers import LayerBase


class DenseLayer(LayerBase):
    """
    A dense layer in a neural network.
    """

    def __init__(
        self,
        input_neurons,
        neurons,
        activation_function="sigmoid",
        initial_weight_range=(-0.1, 0.1),
        weights=None,
        use_bias=True,
    ):
        self._weights = DenseLayer._initialize_weights(
            weights, input_neurons, neurons, initial_weight_range, use_bias
        )

        if activation_function not in ("sigmoid", "tanh", "relu", "linear"):
            raise ValueError("Invalid activation function.")
        self._use_bias = use_bias
        self._neurons = neurons
        self._input_neurons = input_neurons
        self._activation_function = activation_function

    @staticmethod
    def _initialize_weights(
        weights, input_neurons, neurons, initial_weight_range, use_bias
    ):
        if isinstance(input_neurons, int):
            expected_shape = (input_neurons + 1 if use_bias else input_neurons, neurons)

            if weights is not None:
                if weights.shape != expected_shape:
                    raise ValueError(
                        f"Weight matrix must have shape ({expected_shape}), was {weights.shape}"
                    )
                return weights

            if initial_weight_range is not None:
                return np.random.uniform(
                    low=initial_weight_range[0],
                    high=initial_weight_range[1],
                    size=expected_shape,
                )

            return np.random.random_sample(expected_shape)

        if weights is not None:
            return weights

        return np.random.random_sample(input_neurons + (neurons,))

    def _multiply_weights(self, X: np.array):
        return X @ self._weights
        # return np.einsum("i,ij->j", X, self._weights)

    def _apply_activation_function(self, data):
        """
        Applies the current activation function to the data.
        """
        if self._activation_function == "sigmoid":
            return Activation.sigmoid(data)

        if self._activation_function == "tanh":
            return Activation.tanh(data)

        if self._activation_function == "relu":
            return Activation.relu(data)

        if self._activation_function == "linear":
            return Activation.linear(data)

        raise NotImplementedError(
            f"Activation function {self._activation_function} not implemented."
        )

    def _apply_activation_function_derivative(self, data):
        """
        Applies the current activation function to the data.
        """
        if self._activation_function == "sigmoid":
            return Activation.sigmoid_derivative(data)

        if self._activation_function == "tanh":
            return Activation.tanh_derivative(data)

        if self._activation_function == "relu":
            return Activation.relu_derivative(data)

        if self._activation_function == "linear":
            return Activation.linear_derivative(data)

        raise NotImplementedError(
            f"Activation function {self._activation_function} not implemented."
        )

    def _add_bias_neuron_conditionally(self, data):
        if self._use_bias:
            return np.concatenate(([1], data))
        return data

    def _get_weights_excluding_bias(self):
        if self._use_bias:
            return self._weights[1:, :]
        return self._weights

    def forward_pass(self, data):
        """
        Data is a row-vector representing a single test case.
        """
        if len(data.shape) != 1:
            return self._apply_activation_function(
                np.einsum("ijk,ijkl->l", data, self._weights)
            )

        return self._apply_activation_function(
            self._multiply_weights(self._add_bias_neuron_conditionally(data))
        )

    def backward_pass(self, J_L_Y, Y, X):
        """
        Returns a tuple of the weight Jacobian and the Jacobian to pass upstream.
        """
        Diag_J_Y_Sum = self._apply_activation_function_derivative(Y)
        J_Y_Sum = np.diag(Diag_J_Y_Sum)
        # Don't include bias node in Jacobian passed upstream
        J_Y_X = np.dot(J_Y_Sum, self._get_weights_excluding_bias().T)
        J_hat_Y_W = np.outer(self._add_bias_neuron_conditionally(X), Diag_J_Y_Sum)
        J_L_W = J_L_Y * J_hat_Y_W
        J_L_X = np.dot(J_L_Y, J_Y_X)

        return J_L_W, J_L_X

    def update_weights(self, jacobians, learning_rate):
        new_weights = self._weights - learning_rate * np.sum(jacobians, axis=0)

        return DenseLayer(
            input_neurons=self._input_neurons,
            neurons=self._neurons,
            activation_function=self._activation_function,
            weights=new_weights,
            use_bias=self._use_bias,
        )

    def get_weights(self):
        return self._weights
