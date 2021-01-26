import numpy as np


class OutputLayer:
    """
    An output layer.
    """

    def __init__(self, input_neurons, softmax=False):
        self._use_softmax = softmax
        self._input_neurons = input_neurons

    @staticmethod
    def _softmax(X):
        """
        The softmax function.
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def backward_pass(self, output):
        """
        Computes Jacobian.
        """
        if self._use_softmax:
            softmax_jacobian = np.array(
                [
                    np.diag(case_output) - np.outer(case_output, case_output)
                    for case_output in output
                ]
            )
            return tensor

        return output

    def forward_pass(self, data):
        if self._use_softmax:
            return OutputLayer._softmax(data)

        return data