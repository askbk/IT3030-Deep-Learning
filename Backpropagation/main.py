import numpy as np
from NeuralNetwork.Math import Loss
from DatasetFactory import DatasetFactory
from DataUtils import translate_labels_to_neuron_activation
from NetworkFactory import NetworkFactory
from PerformanceDisplay import PerformanceDisplay


def run_image_classification():
    train, validate, test = DatasetFactory.new_dataset("./dataset_config.json")

    translated_validation = translate_labels_to_neuron_activation(validate)
    network, training_performance, validation_performance = NetworkFactory.new_network(
        "./network_config.json"
    ).train(
        translate_labels_to_neuron_activation(train),
        minibatches=500,
        validation_set=translated_validation,
    )
    testing_performance = np.array([])

    PerformanceDisplay.display_performance(
        training_performance, validation_performance, testing_performance
    )


if __name__ == "__main__":
    run_image_classification()