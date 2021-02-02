import random


def randomize_dataset(dataset):
    """
    Randomises order of dataset
    """
    return random.sample(dataset, k=len(dataset))