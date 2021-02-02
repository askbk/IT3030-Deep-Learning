import numpy as np


class Activation:
    @staticmethod
    def sigmoid(X):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def sigmoid_derivative(X):
        return Activation.sigmoid(X) * (1 - Activation.sigmoid(X))

    @staticmethod
    def swish(X):
        return X * Activation.sigmoid(X)

    @staticmethod
    def swish_derivative(X):
        return Activation.swish(X) + Activation.sigmoid(X) * (1 - Activation.swish(X))

    @staticmethod
    def tanh(X):
        """
        Hyperbolic tangent
        """
        return np.tanh(X)

    @staticmethod
    def tanh_derivative(X):
        """
        Hyperbolic tangent
        """
        return 1 - Activation.tanh(X) ** 2

    @staticmethod
    def linear(X):
        """
        Linear function
        """
        return X

    @staticmethod
    def linear_derivative(X):
        return np.ones_like(X)

    @staticmethod
    def relu(X):
        """
        Rectified linear unit function
        """
        return np.maximum(X, 0)

    @staticmethod
    def relu_derivative(X):
        return np.where(X <= 0, 0, 1)


class Loss:
    @staticmethod
    def _mean_squared_error(Y: np.array, Y_hat: np.array):
        """
        Calculates mean squared error.
        """
        return 0.5 * ((Y - Y_hat) ** 2).mean()

    @staticmethod
    def _mean_squared_error_derivative(Y: np.array, Y_hat: np.array):
        return Y_hat - Y