from OutputLayer import OutputLayer
import numpy as np


def test_returns_input_without_softmax():
    data = np.array([[1, 2, 3], [2, 3, 4]])
    output = OutputLayer(input_neurons=3).forward_pass(data)

    assert np.all(np.isclose(output, data))


def test_applies_softmax():
    data = np.array([[1, 2, 3], [0.1, 0.3, 0.9]])
    output = OutputLayer(input_neurons=3, softmax=True).forward_pass(data)
    expected = np.array(
        [[0.09003057, 0.24472847, 0.66524096], [0.22487355, 0.27466117, 0.50046528]]
    )
    assert np.all(np.isclose(output, expected))


# def test_backward_pass_no_softmax():
#     data = np.array([[1, 2, 3, 6, 4], [1, 2, 6, 3, 5]])
#     output = OutputLayer(input_neurons=5, softmax=False).backward_pass()
#     # assert np.all(np.isclose(output, np.diag(np.ones(5))))
