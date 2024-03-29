import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork.Layers import ConvolutionLayer
from DatasetFactory import DatasetFactory
from DataUtils import translate_labels_to_neuron_activation
from NetworkFactory import NetworkFactory
from PerformanceDisplay import PerformanceDisplay


def display_kernels(network):
    for layer in network._layers:
        if isinstance(layer, ConvolutionLayer):
            kernel = layer.get_weights()
            print("showing kernel with shape", kernel.shape)
            c, r, l = kernel.shape
            plt.matshow(kernel.reshape((c * r, l)))
            plt.show()

    kernel = next(
        l for l in network._layers if isinstance(l, ConvolutionLayer)
    ).get_weights()
    c, r, l = kernel.shape
    plt.matshow(kernel.reshape((c * r, l)))
    plt.show()


def run_image_classification_dense():
    train, validate, test = [
        translate_labels_to_neuron_activation(dataset)
        for dataset in DatasetFactory.new_dataset("./configs/1dataset.json")
    ]

    network, training_performance, validation_performance = NetworkFactory.new_network(
        "./configs/1network.json"
    ).train(train, minibatches=100, validation_set=validate)
    testing_performance = network.test(test)

    PerformanceDisplay.display_performance(
        training_performance, validation_performance, testing_performance
    )


def run_image_classification_convolution():
    train, validate, test = [
        translate_labels_to_neuron_activation(dataset)
        for dataset in DatasetFactory.new_dataset("./configs/4dataset.json")
    ]

    network, training_performance, validation_performance = NetworkFactory.new_network(
        "./configs/4network.json"
    ).train(train, 100, validate)
    testing_performance = network.test(test, verbose=False)

    PerformanceDisplay.display_performance(
        training_performance, validation_performance, testing_performance
    )

    display_kernels(network)


if __name__ == "__main__":
    run_image_classification_convolution()
