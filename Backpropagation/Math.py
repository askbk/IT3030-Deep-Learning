import numpy as np


class Activation:
    @staticmethod
    def _sigmoid(X):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _sigmoid_derivative(X):
        return Activation._sigmoid(X) * (1 - Activation._sigmoid(X))

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
        return 1 - Activation._tanh(X) ** 2

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