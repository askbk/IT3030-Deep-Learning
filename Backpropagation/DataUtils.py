import random


def randomize_dataset(X, Y):
    """
    Randomises order of dataset
    """
    samples = len(X)
    XY = list(zip(X, Y))

    randomized_X, randomized_Y = zip(*random.sample(XY, k=samples))

    return list(randomized_X), list(randomized_Y)