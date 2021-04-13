from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras
from Visualization import graph_training_history


class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        # input (28, 28, 1)
        self._layers = [
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
            keras.layers.MaxPooling2D((2, 2), padding="same"),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
            keras.layers.MaxPooling2D((2, 2), padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(units=98),
        ]

    def call(self, inputs):
        return reduce(lambda data, layer: layer(data), self._layers, inputs)


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self._layers = [
            keras.layers.Reshape((7, 7, 2)),
            keras.layers.Conv2DTranspose(
                32, (3, 3), strides=2, activation="relu", padding="same"
            ),
            keras.layers.Conv2DTranspose(
                32, (3, 3), strides=2, activation="relu", padding="same"
            ),
            keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
        ]

    def call(self, inputs):
        return reduce(lambda data, layer: layer(data), self._layers, inputs)


class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self._encoder = Encoder()
        self._decoder = Decoder()

    def call(self, inputs):
        return self._decoder(self._encoder(inputs))

    @staticmethod
    def train(config: dict, unlabeled, return_learning_progress=False):
        autoencoder = Autoencoder()
        autoencoder.compile(loss=config.get("loss"), optimizer=config.get("optimizer"))
        history = autoencoder.fit(
            x=unlabeled,
            y=unlabeled,
            epochs=config.get("epochs"),
            batch_size=config.get("batch_size"),
            shuffle=True,
            validation_split=0.1,
        )
        if return_learning_progress:
            return history
        return autoencoder
