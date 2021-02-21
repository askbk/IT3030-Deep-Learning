import json
from NeuralNetwork import Network
from NeuralNetwork.Layers import OutputLayer, DenseLayer, InputLayer
from functools import reduce


class NetworkFactory:
    """
    Creates a neural network from a config file
    """

    @staticmethod
    def _parse_regularization_config(config: dict):
        if not "regularization" in config:
            return {"regularization": None}

        return {
            "regularization": config["regularization"]["function"],
            "regularization_rate": config["regularization"]["rate"],
        }

    @staticmethod
    def _construct_layers(config: dict):
        def construct_layer(constructed, current_layer_config):
            if len(constructed) == 0:
                return [
                    DenseLayer(
                        input_neurons=config["input_neurons"], **current_layer_config
                    )
                ]

            return [
                *constructed,
                DenseLayer(
                    input_neurons=constructed[-1]._neurons, **current_layer_config
                ),
            ]

        hidden_layers = reduce(
            construct_layer,
            config["layers"],
            [],
        )

        return [
            InputLayer(),
            *hidden_layers,
            OutputLayer(
                input_neurons=hidden_layers[-1]._neurons, softmax=config["softmax"]
            ),
        ]

    @staticmethod
    def new_network(config_path: str):
        """
        Create a new Network using JSON config file.
        """
        with open(config_path, "r") as f:
            config_data = f.read()

        config = json.loads(config_data)

        return Network(
            layers=NetworkFactory._construct_layers(config),
            loss_function=config["loss"],
            **NetworkFactory._parse_regularization_config(config),
            learning_rate=config["learning_rate"],
        )
