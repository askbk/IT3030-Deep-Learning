from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras

from Autoencoder import Encoder


class Classifier(keras.Model):
    def __init__(self, encoder=None):
        super(Classifier, self).__init__()
        self._encoder = Encoder() if encoder is None else encoder
        self._layers = [keras.layers.Dense(10, activation="softmax")]

    def call(self, inputs):
        return reduce(
            lambda data, layer: layer(data), self._layers, self._encoder(inputs)
        )
