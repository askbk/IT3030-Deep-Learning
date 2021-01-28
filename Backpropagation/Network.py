from functools import reduce
import numpy as np
from Math import Loss


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
    def _split_data_into_minibatches(X, Y, minibatches):
        minibatch_size = len(X) // minibatches
        X_batches = [
            X[i : i + minibatch_size] for i in range(0, len(X), minibatch_size)
        ]
        Y_batches = [
            Y[i : i + minibatch_size] for i in range(0, len(Y), minibatch_size)
        ]
        return zip(X_batches, Y_batches)

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

    def _apply_loss_function(self, Y, Y_hat):
        """
        docstring
        """
        if self._loss_function == "mse":
            return Loss._mean_squared_error(Y, Y_hat)

        raise NotImplementedError

    def _apply_loss_function_derivative(self, Y, Y_hat):
        if self._loss_function == "mse":
            return Loss._mean_squared_error_derivative(Y, Y_hat)

        raise NotImplementedError

    def _forward_backward(self, case):
        """
        Performs a forward pass, followed by a backward pass.
        Returns the weight jacobian for each layer.
        """
        x, y = case
        # compute outputs at each layer
        layer_outputs = self._cached_forward_pass(x)
        # compute loss jacobian
        loss_jacobian = self._apply_loss_function_derivative(layer_outputs[-1], y)
        # print(case, layer_outputs[-1], loss_jacobian)
        # backpropagate
        weight_jacobians = list()
        downstream_jacobian = loss_jacobian
        # print(layer_outputs, len(layer_outputs))
        for index in reversed(range(len(self._layers))):
            # print(f"layer {index} of {len(self._layers) - 1}")
            weight_jacobian, pass_down = self._layers[index].backward_pass(
                downstream_jacobian, layer_outputs[index], layer_outputs[index - 1]
            )
            weight_jacobians.append(weight_jacobian)
            downstream_jacobian = pass_down

        return reversed(weight_jacobians)

    def _train_minibatch(self, batch):
        X, Y = batch

        weight_jacobians = map(lambda case: self._forward_backward(case), zip(X, Y))

        updated_layers = map(
            lambda layer_jacobians: layer_jacobians[0].update_weights(
                layer_jacobians[1], self._learning_rate
            ),
            zip(self._layers, zip(*weight_jacobians)),
        )

        return Network._update_layers(self, list(updated_layers))

    def train(self, X, Y, minibatches):
        """
        Returns a new instance of the Network that is trained with X and Y.
        """
        return reduce(
            lambda network, batch: network._train_minibatch(batch),
            Network._split_data_into_minibatches(X, Y, minibatches),
            self,
        )
