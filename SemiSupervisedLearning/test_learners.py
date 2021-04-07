from tensorflow import keras
from tensorflow.keras import datasets


from Autoencoder import Autoencoder
from Classifier import Classifier
from Utils import semisupervised_split, preprocess, display


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


def test_autoencoder():
    (train_data, _), (test_data, _) = get_preprocessed_data("mnist")

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
    (x_train, y_train), (x_test, y_test) = get_preprocessed_data("mnist")

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
    (x1_train, x2_train, y2_train), (x1_test, x2_test, y2_test) = get_preprocessed_data(
        "mnist", semi_supervised_split=True, labeled_fraction=0.5
    )
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
