import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets


def semisupervised_split(dataset, labeled_fraction=0.1):
    x, y = dataset
    split_index = int(len(x) * (1 - labeled_fraction))
    return (x[:split_index], y[:split_index]), (x[split_index:], y[split_index:])


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

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


def load_dataset(dataset_name: str):
    if dataset_name == "mnist":
        return datasets.mnist.load_data()


def get_preprocessed_data(
    dataset: str, semi_supervised_split=False, labeled_fraction: float = None
):
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    if not semi_supervised_split:
        return (x_train, y_train), (x_test, y_test)

    (x_train_unlabeled, _), (x_train_labeled, y_train_labeled) = semisupervised_split(
        (x_train, y_train), labeled_fraction
    )
    (x_test_unlabeled, _), (x_test_labeled, y_test_labeled) = semisupervised_split(
        (x_test, y_test), labeled_fraction
    )

    return (x_train_unlabeled, x_train_labeled, y_train_labeled), (
        x_test_unlabeled,
        x_test_labeled,
        y_test_labeled,
    )
