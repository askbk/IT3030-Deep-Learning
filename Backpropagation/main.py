import numpy as np
from NeuralNetwork.Math import Loss
from ImageGenerator import ImageGenerator
from DataUtils import translate_labels_to_neuron_activation
from NetworkFactory import NetworkFactory


def run_image_classification():
    train, validate = ImageGenerator.generate(
        side_length=10,
        flatten=True,
        image_set_size=1000,
        image_set_fractions=(0.9, 0.1),
    )

    network = NetworkFactory.new_network("./network_config.json").train(
        translate_labels_to_neuron_activation(train), minibatches=100
    )
    translated_validation = translate_labels_to_neuron_activation(validate)

    validation_output = [
        (
            Y,
            network.forward_pass(X),
            Loss.cross_entropy(Y, network.forward_pass(X)),
        )
        for X, Y in translated_validation
    ]
    validation_error = np.mean(
        [
            Loss.cross_entropy(Y, network.forward_pass(X))
            for X, Y in translated_validation
        ]
    )

    print(f"avg validation error: {validation_error}")
    print(validation_output[:5])
    print([Y for X, Y in validate[:5]])


if __name__ == "__main__":
    run_image_classification()