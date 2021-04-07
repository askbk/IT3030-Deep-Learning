from tensorflow import keras
from tensorflow.keras import datasets


from Autoencoder import Autoencoder
from Classifier import Classifier
from Utils import semisupervised_split, preprocess, display


def load_dataset(dataset_name: str):
    if dataset_name == "mnist":
        return datasets.mnist.load_data()


def get_preprocessed_data(dataset: str):
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
    return (preprocess(x_train), keras.utils.to_categorical(y_train)), (
        preprocess(x_test),
        keras.utils.to_categorical(y_test),
    )


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
    test_supervised_classifier()
