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

    @staticmethod
    def _conditional_flatten(image_set, flatten):
        """
        Flattens the image set if flatten is True.
        """
        if not flatten:
            return image_set

        return list(map(lambda image: list(chain.from_iterable(image)), image_set))

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
        if round(sum(image_set_fractions), 5) != 1:
            raise ValueError("Image set fractions must sum to 1.")

        training, validation, test = image_set_fractions

        if not (0 <= training <= 1 and 0 <= validation <= 1 and 0 <= test <= 1):
            raise ValueError("All image set fractions must be numbers between 0 and 1.")

        image_set = [[[0] * side_length] * side_length] * image_set_size
        training_index = int(training * image_set_size)
        validation_index = int((training + validation) * image_set_size)

        training_set = image_set[:training_index]
        validation_set = image_set[training_index : validation_index + 1]
        test_set = image_set[validation_index + 1 :]

        return (
            ImageGenerator._conditional_flatten(training_set, flatten),
            ImageGenerator._conditional_flatten(validation_set, flatten),
            ImageGenerator._conditional_flatten(test_set, flatten),
        )
