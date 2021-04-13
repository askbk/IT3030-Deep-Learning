import numpy as np
from tensorflow.keras import datasets
from Autoencoder import Autoencoder
from Classifier import Classifier
from Utils import semisupervised_split, preprocess, get_preprocessed_data
from Visualization import graph_training_history, display, tsne_plot


def test_autoencoder():
    (train_data, _), (x_test, y_test) = get_preprocessed_data("mnist")
    autoencoder, history = Autoencoder.train(
        {
            "epochs": 3,
            "batch_size": 256,
            "optimizer": "adam",
            "loss": "mean_squared_error",
        },
        train_data,
        return_learning_progress=True,
    )
    graph_training_history([("autoencoder", history)], keys=["loss"])
    predictions = autoencoder.predict(x_test[:100])
    display(x_test, predictions, n=20)
    encoder_output = autoencoder._encoder(x_test[:1000])
    tsne_plot(np.squeeze(encoder_output), y_test[:1000])


def test_supervised_classifier():
    (x_train, y_train), (x_test, y_test) = get_preprocessed_data("mnist")
    classifier = Classifier.train_supervised(
        {
            "epochs": 3,
            "batch_size": 256,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
        },
        (x_train[:10000], y_train[:10000]),
    )
    # score = classifier.evaluate(x_test, y_test, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])


def test_semi_supervised_classifier():
    (x1_train, x2_train, y2_train), (x1_test, x2_test, y2_test) = get_preprocessed_data(
        "mnist", semi_supervised_split=True, labeled_fraction=0.1
    )
    classifier = Classifier.train_semisupervised(
        {
            "epochs": 1,
            "batch_size": 256,
            "optimizer": "adam",
            "loss": "mean_squared_error",
        },
        {
            "epochs": 3,
            "batch_size": 32,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "freeze_encoder": False,
        },
        x1_train[:10000],
        (x2_train[:10000], y2_train[:10000]),
    )
    # score = classifier.evaluate(x2_test, y2_test, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])


if __name__ == "__main__":
    test_autoencoder()
    # test_supervised_classifier()
    # test_semi_supervised_classifier()
