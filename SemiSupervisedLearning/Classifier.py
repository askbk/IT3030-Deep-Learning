from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras
from Utils import get_optimizer
from Visualization import graph_training_history
from Autoencoder import Encoder, Autoencoder


class ClassifierHead(keras.Model):
    def __init__(self):
        super(ClassifierHead, self).__init__()
        self._layers = [keras.layers.Dense(10, activation="softmax")]

    def call(self, inputs):
        return reduce(lambda data, layer: layer(data), self._layers, inputs)


class Classifier(keras.Model):
    def __init__(
        self, encoder=None, classifier_head=None, freeze_encoder_weights=False
    ):
        super(Classifier, self).__init__()
        self._encoder = Encoder() if encoder is None else encoder
        if freeze_encoder_weights:
            self._encoder.freeze_weights()
        self._classifier_head = (
            ClassifierHead() if classifier_head is None else classifier_head
        )

    def call(self, inputs):
        return self._classifier_head(self._encoder(inputs))

    @staticmethod
    def train_supervised(config: dict, training_set, return_learning_progress=False):
        classifier = Classifier()
        classifier.compile(
            loss=config.get("loss"),
            optimizer=get_optimizer(
                config.get("optimizer", "adam"), config.get("learning_rate", 0.001)
            ),
            metrics=["accuracy"],
        )
        x, y = training_set
        history = classifier.fit(
            x,
            y,
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            validation_split=0.1,
        )
        if return_learning_progress:
            return history

        return classifier

    @staticmethod
    def train_semisupervised(
        autoencoder_config: dict,
        classifier_config: dict,
        unlabeled_training_set,
        labeled_training_set,
        return_learning_progress=False,
    ):
        autoencoder = Autoencoder.train(autoencoder_config, unlabeled_training_set)
        classifier = Classifier(
            encoder=autoencoder._encoder,
            freeze_encoder_weights=classifier_config.get("freeze_encoder", False),
        )
        classifier.compile(
            loss=classifier_config.get("loss"),
            optimizer=get_optimizer(
                config.get("optimizer", "adam"), config.get("learning_rate", 0.001)
            ),
            metrics=["accuracy"],
        )

        x, y = labeled_training_set

        history = classifier.fit(
            x,
            y,
            batch_size=classifier_config.get("batch_size"),
            epochs=classifier_config.get("epochs"),
            validation_split=0.1,
        )
        if return_learning_progress:
            return history
        return classifier
