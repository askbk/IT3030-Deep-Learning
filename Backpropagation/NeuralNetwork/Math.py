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
    def sigmoid_derivative(Y):
        return Y * (1.0 - Y)

    @staticmethod
    def tanh(X):
        """
        Hyperbolic tangent
        """
        return np.tanh(X)

    @staticmethod
    def tanh_derivative(Y):
        """
        Hyperbolic tangent
        """
        return 1 - Y ** 2

    @staticmethod
    def linear(X):
        """
        Linear function
        """
        return X

    @staticmethod
    def linear_derivative(Y):
        return np.ones_like(Y)

    @staticmethod
    def relu(X):
        """
        Rectified linear unit function
        """
        return np.maximum(X, 0)

    @staticmethod
    def relu_derivative(Y):
        return np.where(Y <= 0, 0, 1)

    @staticmethod
    def softmax(X):
        """
        The softmax function.
        """
        return softmax(X, axis=0)

    @staticmethod
    def softmax_derivative(S):
        S_vector = S.reshape(S.shape[0], 1)
        S_matrix = np.tile(S_vector, S.shape[0])
        return np.diag(S) - (S_matrix * np.transpose(S_matrix))


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


def glorot_init(input_neurons, layer_neurons, weight_count):
    bound = np.sqrt(6 / (input_neurons + layer_neurons))
    return np.random.default_rng().uniform(-bound, bound, weight_count)
