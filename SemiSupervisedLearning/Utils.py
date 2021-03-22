import numpy as np


def semisupervised_split(dataset, labeled_fraction=0.1):
    x, y = dataset
    split_index = int(len(x) * (1 - labeled_fraction))
    return (np.expand_dims(x[:split_index], -1), np.expand_dims(y[:split_index], -1)), (
        np.expand_dims(x[split_index:], -1),
        np.expand_dims(y[split_index:], -1),
    )
