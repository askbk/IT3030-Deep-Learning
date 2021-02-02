import numpy as np
from NeuralNetwork.Network import Network
from NeuralNetwork.Layer import Layer
from NeuralNetwork.InputLayer import InputLayer
from NeuralNetwork.OutputLayer import OutputLayer
from NeuralNetwork.Math import Loss
from ImageGenerator import ImageGenerator
from DataUtils import translate_labels_to_neuron_activation


def run_image_classification():
    train, validate, _ = ImageGenerator.generate(
        side_length=10, flatten=True, image_set_size=1000
    )

    network = Network(
        layers=[
            InputLayer(),
            Layer(input_neurons=100, neurons=1000),
            Layer(input_neurons=1000, neurons=1000),
            Layer(input_neurons=1000, neurons=500),
            Layer(input_neurons=500, neurons=4),
            OutputLayer(input_neurons=4, softmax=True),
        ],
        loss_function="cross_entropy",
        regularization=None,
        learning_rate=0.01,
    ).train(translate_labels_to_neuron_activation(train), minibatches=10)
    validation_error = np.mean(
        [
            Loss.cross_entropy(Y, network.forward_pass(X))
            for X, Y in translate_labels_to_neuron_activation(validate)
        ]
    )
    print(f"avg validation error: {validation_error}")


if __name__ == "__main__":
    run_image_classification()