import numpy as np
from Backpropagation.NeuralNetwork.Math import Activation


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
        use_bias=True,
    ):
        self._weights = Layer._initialize_weights(
            weights, input_neurons, neurons, initial_weight_range, use_bias
        )

        if activation_function not in ("sigmoid", "tanh", "relu", "linear", "swish"):
            raise ValueError("Invalid activation function.")
        self._use_bias = use_bias
        self._neurons = neurons
        self._input_neurons = input_neurons
        self._activation_function = activation_function

    @staticmethod
    def _initialize_weights(
        weights, input_neurons, neurons, initial_weight_range, use_bias
    ):
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

        return np.random.rand(expected_shape)

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

        if self._activation_function == "swish":
            return Activation.swish(data)

        raise NotImplementedError()

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

        if self._activation_function == "swish":
            return Activation.swish_derivative(data)

        raise NotImplementedError()

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
            self._multiply_weights(self._add_bias_neuron_conditionally(data))
        )

    def backward_pass(self, J_L_Z, Z, Y):
        """
        Returns a tuple of the weight Jacobian and the Jacobian to pass upstream.
        """
        Diag_J_Z_Sum = self._apply_activation_function_derivative(Z)
        J_Z_Sum = np.diag(Diag_J_Z_Sum)
        # Don't include bias node in Jacobian passed upstream
        J_Z_Y = np.dot(J_Z_Sum, self._get_weights_excluding_bias().T)
        # J_hat_Z_W = np.outer(Diag_J_Z_Sum, self._add_bias_neuron_conditionally(Y))
        J_hat_Z_W = np.outer(self._add_bias_neuron_conditionally(Y), Diag_J_Z_Sum)
        J_L_W = J_L_Z * J_hat_Z_W
        J_L_Y = np.dot(J_L_Z, J_Z_Y)

        return J_L_W, J_L_Y

    def update_weights(self, jacobians, learning_rate):
        new_weights = self._weights - learning_rate * np.sum(jacobians, axis=0)

        return Layer(
            input_neurons=self._input_neurons,
            neurons=self._neurons,
            activation_function=self._activation_function,
            weights=new_weights,
            use_bias=self._use_bias,
        )
