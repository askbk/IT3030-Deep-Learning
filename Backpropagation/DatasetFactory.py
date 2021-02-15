import json
from ImageGenerator import ImageGenerator


class DatasetFactory:
    @staticmethod
    def new_dataset(config_path):
        """
        Creates a dataset from a JSON config file
        """
        with open(config_path, "r") as f:
            config_data = f.read()

        config = json.loads(config_data)

        return ImageGenerator().generate(**config)
