import numpy as np
from Utils import semisupervised_split


def test_semisupervised_split():
    x_case_1 = np.ones(6).reshape((2, 3, 1))
    x_case_2 = np.zeros(6).reshape((2, 3, 1))
    x_case_3 = np.full((2, 3, 1), 2)
    x_case_4 = np.full((2, 3, 1), 3)
    y_case_1 = 1
    y_case_2 = 0
    y_case_3 = 2
    y_case_4 = 3
    data = (
        np.array([x_case_1, x_case_2, x_case_3, x_case_4]),
        np.array([y_case_1, y_case_2, y_case_3, y_case_4]),
    )

    (x1, y1), (x2, y2) = semisupervised_split(data, labeled_fraction=0.5)
    print(x1.shape, y1.shape, x2.shape, y2.shape)
    assert len(x1) == len(y1) == 2
    assert len(x2) == len(y2) == 2
    assert np.all(np.equal(x1, [x_case_1, x_case_2]))
    assert np.all(np.equal(y1, [y_case_1, y_case_2]))
    assert np.all(np.equal(x2, [x_case_3, x_case_4]))
    assert np.all(np.equal(y2, [y_case_3, y_case_4]))
