import numpy as np
from NeuralNetwork.Layers import ConvolutionLayer
from DatasetFactory import DatasetFactory
from DataUtils import translate_labels_to_neuron_activation
from NetworkFactory import NetworkFactory
from PerformanceDisplay import PerformanceDisplay


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
    # print(
    #     next(
    #         l for l in network._layers if isinstance(l, ConvolutionLayer)
    #     ).get_weights()
    # )


if __name__ == "__main__":
    run_image_classification_convolution()
