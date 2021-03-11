import operator
import numpy as np
from functools import reduce
from NeuralNetwork.Math import Activation, glorot_init
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
        weights, input_shape, neurons, initial_weight_range, use_bias
    ):
        input_neurons = (
            np.product(input_shape) + 1 if use_bias else np.product(input_shape)
        )
        expected_shape = (input_neurons, neurons)

        if weights is not None:
            if weights.shape != expected_shape:
                raise ValueError(
                    f"Weight matrix must have shape ({expected_shape}), was {weights.shape}"
                )
            return weights

        if initial_weight_range is None:
            return glorot_init(input_neurons, neurons, input_neurons * neurons).reshape(
                expected_shape
            )

        if initial_weight_range is not None:
            return np.random.uniform(
                low=initial_weight_range[0],
                high=initial_weight_range[1],
                size=expected_shape,
            )

        return np.random.random_sample(expected_shape)

    def _multiply_weights(self, X: np.array):
        return X @ self._weights

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
        return self._apply_activation_function(
            self._multiply_weights(self._add_bias_neuron_conditionally(data.flatten()))
        )

    def backward_pass(self, J_L_Y, Y, X):
        """
        Returns a tuple of the weight Jacobian and the Jacobian to pass upstream.
        """
        Diag_J_Y_Sum = self._apply_activation_function_derivative(Y)
        flattened_X = X.flatten()
        J_Y_Sum = np.diag(Diag_J_Y_Sum)
        J_hat_Y_W = np.outer(
            self._add_bias_neuron_conditionally(flattened_X), Diag_J_Y_Sum
        )
        J_Y_X = np.dot(J_Y_Sum, self._get_weights_excluding_bias().T)
        J_L_X = np.dot(J_L_Y, J_Y_X)
        J_L_W = J_L_Y * J_hat_Y_W

        return J_L_W, J_L_X.reshape(X.shape)

    def update_weights(self, jacobians, learning_rate):
        jacobian_sum = reduce(np.add, jacobians)
        new_weights = self._weights - learning_rate * jacobian_sum

        return DenseLayer(
            input_neurons=self._input_neurons,
            neurons=self._neurons,
            activation_function=self._activation_function,
            weights=new_weights,
            use_bias=self._use_bias,
        )

    def get_weights(self):
        return self._weights

    def __repr__(self):
        return f"DenseLayer<neurons={self._neurons}, activation={self._activation_function}, use_bias={self._use_bias}>"
