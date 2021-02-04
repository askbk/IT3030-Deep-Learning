from functools import reduce
import numpy as np
from NeuralNetwork.Math import Loss


class Network:
    """
    A neural network.
    """

    def __init__(
        self,
        layers,
        loss_function,
        regularization,
        regularization_rate=None,
        learning_rate=0.1,
    ):
        self._layers = layers
        self._loss_function = loss_function
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._regularization_rate = regularization_rate

    @staticmethod
    def _update_layers(old, layers):
        """
        Returns a new instance of old with new layers.
        """
        return Network(
            layers=layers,
            loss_function=old._loss_function,
            regularization=None,
            regularization_rate=None,
            # regularization=old._regularization,
            # regularization_rate=old._regularization_rate,
        )

    @staticmethod
    def _split_data_into_minibatches(dataset, minibatches):
        minibatch_size = len(dataset) // minibatches
        return [
            dataset[i : i + minibatch_size]
            for i in range(0, len(dataset), minibatch_size)
        ]

    def forward_pass(self, X):
        """
        Forward pass through the network.
        """
        return reduce(lambda data, layer: layer.forward_pass(data), self._layers, X)

    def _cached_forward_pass(self, X):
        """
        Forward pass through the network, caching the output from each layer
        """
        return reduce(
            lambda results, layer: [*results, layer.forward_pass(results[-1])],
            self._layers[1:],
            [X],
        )

    def _apply_penalty_function(self):
        if self._regularization is None:
            return 0

        weights = np.concatenate(
            [layer.get_weights().flatten() for layer in self._layers]
        )

        if self._regularization == "l1":
            return Loss.L1_regularization(weights) * self._regularization_rate

        if self._regularization == "l2":
            return Loss.L2_regularization(weights) * self._regularization_rate

        raise NotImplementedError

    def _apply_regularization_derivative(self, weights):
        if self._regularization is None or weights is None:
            return np.zeros_like(weights)

        if self._regularization == "l1":
            return self._regularization_rate * np.sign(weights)

        if self._regularization == "l2":
            return self._regularization_rate * weights

        raise NotImplementedError

    def _apply_loss_function(self, Y, Y_hat):
        if self._loss_function == "mse":
            return Loss.mean_squared_error(Y, Y_hat) + self._apply_penalty_function()

        if self._loss_function == "cross_entropy":
            return Loss.cross_entropy(Y, Y_hat) + self._apply_penalty_function()

        raise NotImplementedError

    def _apply_loss_function_derivative(self, Y, Y_hat):
        if self._loss_function == "mse":
            return Loss.mean_squared_error_derivative(Y, Y_hat)

        if self._loss_function == "cross_entropy":
            return Loss.cross_entropy_derivative(Y, Y_hat)

        raise NotImplementedError

    def _forward_backward(self, case):
        """
        Performs a forward pass, followed by a backward pass.
        Returns the weight jacobian for each layer.
        """
        x, y = case
        layer_outputs = self._cached_forward_pass(np.array(x, dtype=np.float64))
        loss_jacobian = np.array(
            [
                self._apply_loss_function_derivative(
                    np.array(y, dtype=np.float64), layer_outputs[-1]
                )
            ]
        )
        weight_jacobians = list()
        downstream_jacobian = loss_jacobian
        for index in reversed(range(len(self._layers))):
            weight_jacobian, pass_down = self._layers[index].backward_pass(
                downstream_jacobian, layer_outputs[index], layer_outputs[index - 1]
            )
            weight_jacobians.append(weight_jacobian)
            downstream_jacobian = pass_down

        return reversed(weight_jacobians)

    def _train_minibatch(self, batch):
        weight_jacobians = map(lambda case: self._forward_backward(case), batch)
        regularization_jacobians = map(
            lambda layer: self._apply_regularization_derivative(layer.get_weights()),
            self._layers,
        )

        def update_layer_weights(args):
            layer, weight_jac, reg_jac = args
            return layer.update_weights(
                list(weight_jac + (reg_jac,)), self._learning_rate
            )

        updated_layers = list(
            map(
                update_layer_weights,
                zip(self._layers, zip(*weight_jacobians), regularization_jacobians),
            )
        )

        return Network._update_layers(self, updated_layers)

    def train(self, dataset, minibatches):
        """
        Returns a new instance of the Network that is trained with X and Y.
        """
        return reduce(
            lambda network, batch: network._train_minibatch(batch),
            Network._split_data_into_minibatches(dataset, minibatches),
            self,
        )
