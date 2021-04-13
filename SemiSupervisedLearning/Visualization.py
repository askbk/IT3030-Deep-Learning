import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import Sequence, Tuple
from itertools import chain


def graph_training_history(
    histories: Sequence[Tuple[str, tf.keras.callbacks.History]],
    keys: Sequence[str],
    title: str = None,
):
    plt.figure(1)
    keys_to_use = list(
        set(
            filter(
                lambda key: any(filter_key in key for filter_key in keys),
                chain.from_iterable(history.history.keys() for _, history in histories),
            )
        )
    )
    legend_keys = []
    for name, history in histories:
        for key in history.history.keys():
            if key in keys_to_use:
                plt.plot(history.history[key])
                legend_keys.append(f"{name} {key}")
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend(legend_keys, loc="upper left")
    plt.show()


def display(array1, array2, n=10):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
