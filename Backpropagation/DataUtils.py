import random
import numpy as np
from functools import reduce


def randomize_dataset(dataset):
    """
    Randomises order of dataset
    """
    return random.sample(dataset, k=len(dataset))


def translate_labels_to_neuron_activation(dataset):
    labels = reduce(lambda labels_set, case: labels_set | {case[1]}, dataset, set())
    neuron_dict = {label: i for i, label in enumerate(labels)}
    neuron_count = len(labels)

    def translated(label):
        return np.array(
            [1 if neuron_dict[label] == x else 0 for x in range(neuron_count)],
            dtype=np.float64,
        )

    return list(map(lambda case: (case[0], translated(case[1])), dataset))
