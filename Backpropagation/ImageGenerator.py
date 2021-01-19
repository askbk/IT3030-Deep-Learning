from itertools import chain
import random
import math
import numpy as np
from itertools import product


class ImageGenerator:
    """
    Static class for generating images.
    """

    @staticmethod
    def _get_relative_tolerance(figure_size):
        """
        Relative tolerance to use when drawing figures
        """
        return 0.3 / math.log2(figure_size)

    @staticmethod
    def _calculate_figure_center(side_length, figure_size, centered):
        """
        Calculate the center for a figure.
        """
        half = side_length / 2

        if centered:
            return half, half

        wiggle = (side_length - figure_size) / 2

        return (
            half + random.uniform(-wiggle, wiggle),
            half + random.uniform(-wiggle, wiggle),
        )

    @staticmethod
    def _distance(point_a, point_b):
        """
        Calculate distance between points in 2D
        """
        x1, y1 = point_a
        x2, y2 = point_b

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def _generate_circle(side_length, figure_size, centered):
        """
        Generates a circle
        """
        center = ImageGenerator._calculate_figure_center(
            side_length, figure_size, centered
        )

        image = np.zeros((side_length, side_length))

        for x, y in product(range(side_length), range(side_length)):
            center_distance = ImageGenerator._distance((x, y), center)
            if math.isclose(
                figure_size / 2,
                center_distance,
                rel_tol=ImageGenerator._get_relative_tolerance(figure_size),
            ):
                image[x, y] = 1
        return image

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
        centered=False,
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
        image_set = [
            ImageGenerator._generate_circle(
                side_length=side_length,
                figure_size=random.randint(5, 50),
                centered=centered,
            )
            for i in range(image_set_size)
        ]
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


if __name__ == "__main__":
    from ImageViewer import ImageViewer

    images = ImageGenerator().generate(side_length=50, centered=False)[0]

    ImageViewer.display_images(images)