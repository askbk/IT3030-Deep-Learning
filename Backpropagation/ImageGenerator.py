from itertools import chain


class ImageGenerator:
    """
    Class for generating images.
    """

    def __init__(
        self,
        side_length=10,
        noise=0,
        figure_size_range=(7, 10),
        flatten=False,
    ):
        pass

    def generate(
        self,
        image_set_size=100,
        image_set_fractions=(0.7, 0.2, 0.1),
        side_length=10,
        flatten=False,
    ):
        """
        Generates images.
        """
        training, validation, test = image_set_fractions

        image_set = [[[0] * side_length] * side_length] * image_set_size
        training_index = int(training * image_set_size)
        validation_index = int((training + validation) * image_set_size)

        training_set = image_set[:training_index]
        validation_set = image_set[training_index : validation_index + 1]
        test_set = image_set[validation_index + 1 :]

        if not flatten:
            return (
                training_set,
                validation_set,
                test_set,
            )

        return (
            list(map(lambda image: list(chain.from_iterable(image)), training_set)),
            list(map(lambda image: list(chain.from_iterable(image)), validation_set)),
            list(map(lambda image: list(chain.from_iterable(image)), test_set)),
        )
