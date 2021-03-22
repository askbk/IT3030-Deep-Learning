import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.models import Model

from Autoencoder import Autoencoder
from Classifier import Classifier
from Utils import semisupervised_split


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


def test_autoencoder():
    (train_data, _), (test_data, _) = datasets.mnist.load_data()

    # Normalize and reshape the data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    autoencoder = Autoencoder()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(
        x=train_data,
        y=train_data,
        epochs=1,
        batch_size=128,
        shuffle=True,
        validation_data=(test_data, test_data),
    )
    predictions = autoencoder.predict(test_data)
    display(test_data, predictions)


def test_supervised_classifier():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    classifier = Classifier()
    batch_size = 128
    epochs = 1

    classifier.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    classifier.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    score = classifier.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def test_semi_supervised_classifier():
    labeled_fraction = 0.5
    train, test = datasets.mnist.load_data()
    (x1_train, _), (x2_train, y2_train) = semisupervised_split(train, labeled_fraction)
    (x1_test, _), (x2_test, y2_test) = semisupervised_split(test, labeled_fraction)

    # Normalize and reshape the data
    x1_train = preprocess(x1_train)
    x1_test = preprocess(x1_test)
    x2_train = preprocess(x2_train)
    x2_test = preprocess(x2_test)

    autoencoder = Autoencoder()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(
        x=x1_train,
        y=x1_train,
        epochs=1,
        batch_size=128,
        shuffle=True,
        validation_data=(x1_test, x1_test),
    )

    num_classes = 10

    y2_train = keras.utils.to_categorical(y2_train, num_classes)
    y2_test = keras.utils.to_categorical(y2_test, num_classes)

    classifier = Classifier(encoder=autoencoder._encoder)
    batch_size = 128
    epochs = 1

    classifier.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    classifier.fit(
        x2_train, y2_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    score = classifier.evaluate(x2_test, y2_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    test_semi_supervised_classifier()
