import numpy as np
from NeuralNetwork.Math import Loss
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
        for dataset in DatasetFactory.new_dataset("./configs/5dataset.json")
    ]

    network = NetworkFactory.new_network("./configs/5network.json")
    # training_performance = network.test(train)
    # validation_performance = network.test(validate)
    testing_performance = network.test(test, verbose=False)

    PerformanceDisplay.display_performance(
        np.ones(100), np.ones(100), testing_performance
    )


if __name__ == "__main__":
    run_image_classification_convolution()
