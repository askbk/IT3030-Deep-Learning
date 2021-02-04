import numpy as np
from scipy.special import expit, softmax


class Activation:
    @staticmethod
    def sigmoid(X):
        """
        Sigmoid function
        """
        return expit(X)

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

    @staticmethod
    def softmax(X):
        """
        The softmax function.
        """
        return softmax(X, axis=0)

    @staticmethod
    def softmax_derivative(X):
        X_reshape = X.reshape((-1, 1))
        return np.diagflat(X) - np.dot(X_reshape, X_reshape.T)


class Loss:
    @staticmethod
    def mean_squared_error(Y: np.array, Y_hat: np.array):
        """
        Calculates mean squared error.
        """
        return 0.5 * ((Y - Y_hat) ** 2).mean()

    @staticmethod
    def mean_squared_error_derivative(Y: np.array, Y_hat: np.array):
        return Y_hat - Y

    @staticmethod
    def cross_entropy(Y: np.array, Y_hat: np.array):
        return -np.dot(Y, np.log(Y_hat))

    @staticmethod
    def cross_entropy_derivative(Y: np.array, Y_hat: np.array):
        return -Y / Y_hat

    @staticmethod
    def L1_regularization(parameters):
        return np.sum(np.abs(parameters))

    @staticmethod
    def L2_regularization(parameters):
        return 0.5 * np.sum(parameters ** 2)
