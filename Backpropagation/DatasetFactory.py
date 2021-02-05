import json
from ImageGenerator import ImageGenerator


class DatasetFactory:
    """
    Creates a dataset from a config file
    """

    @staticmethod
    def new_dataset(config_path):
        with open(config_path, "r") as f:
            config_data = f.read()

        config = json.loads(config_data)

        return ImageGenerator().generate(**config)
