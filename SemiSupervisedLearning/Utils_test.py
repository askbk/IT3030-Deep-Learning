import numpy as np
from Utils import semisupervised_split


def test_semisupervised_split():
    data = (np.ones(90).reshape((10, 3, 3)), np.arange(10))
    (x1, y1), (x2, y2) = semisupervised_split(data, labeled_fraction=0.5)
    assert len(x1) == len(y1) == 5
    assert len(x2) == len(y2) == 5
