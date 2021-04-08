from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras

from Autoencoder import Encoder, Autoencoder


class ClassifierHead(keras.Model):
    def __init__(self):
        super(ClassifierHead, self).__init__()
        self._layers = [keras.layers.Dense(10, activation="softmax")]

    def call(self, inputs):
        return reduce(lambda data, layer: layer(data), self._layers, inputs)


class Classifier(keras.Model):
    def __init__(self, encoder=None, classifier_head=None):
        super(Classifier, self).__init__()
        self._encoder = Encoder() if encoder is None else encoder
        self._classifier_head = (
            ClassifierHead() if classifier_head is None else classifier_head
        )

    def call(self, inputs):
        return self._classifier_head(self._encoder(inputs))

    @staticmethod
    def train_supervised(config: dict, training_set):
        classifier = Classifier()
        classifier.compile(
            loss=config.get("loss"),
            optimizer=config.get("optimizer"),
            metrics=["accuracy"],
        )
        x, y = training_set
        classifier.fit(
            x,
            y,
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            validation_split=0.1,
        )
        return classifier

    @staticmethod
    def train_semisupervised(
        autoencoder_config: dict,
        classifier_config: dict,
        unlabeled_training_set,
        labeled_training_set,
    ):
        autoencoder = Autoencoder.train(autoencoder_config, unlabeled_training_set)
        classifier = Classifier(encoder=autoencoder._encoder)
        classifier.compile(
            loss=classifier_config.get("loss"),
            optimizer=classifier_config.get("optimizer"),
            metrics=["accuracy"],
        )

        x, y = labeled_training_set

        classifier.fit(
            x,
            y,
            batch_size=classifier_config.get("batch_size"),
            epochs=classifier_config.get("epochs"),
            validation_split=0.1,
        )

        return classifier
