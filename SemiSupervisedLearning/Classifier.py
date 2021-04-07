from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras

from Autoencoder import Encoder


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
