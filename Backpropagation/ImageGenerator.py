from itertools import chain
import random
import math
import numpy as np
from itertools import product
from typing import Tuple


class ImageGenerator:
    """
    Static class for generating images.
    """

    @staticmethod
    def _get_relative_tolerance(figure_size: int):
        """
        Relative tolerance to use when drawing figures
        """
        return 0.3 / math.log2(figure_size)

    @staticmethod
    def _calculate_figure_center(side_length: int, figure_size: int, centered: bool):
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
    def _distance(point_a: Tuple[int], point_b: Tuple[int]):
        """
        Calculate distance between points in 2D
        """
        x1, y1 = point_a
        x2, y2 = point_b

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def _add_noise(image, noise):
        """
        Return the image with added noise
        """
        return [
            [pixel if random.random() > noise else 1 - pixel for pixel in row]
            for row in image
        ]

    @staticmethod
    def _generate_circle(side_length: int, figure_size: int, center: Tuple[float]):
        """
        Generates a circle
        """
        tolerance = ImageGenerator._get_relative_tolerance(figure_size)

        def should_be_colored(x, y):
            center_distance = ImageGenerator._distance((x, y), center)
            return math.isclose(
                figure_size / 2,
                center_distance,
                rel_tol=tolerance,
            )

        return list(
            map(
                lambda x: list(
                    map(
                        lambda y: 1 if should_be_colored(x, y) else 0,
                        range(side_length),
                    )
                ),
                range(side_length),
            ),
        )

    @staticmethod
    def _generate_cross(side_length: int, figure_size: int, center: Tuple[float]):
        """
        Generates a cross
        """
        image = np.zeros((side_length, side_length))
        tolerance = ImageGenerator._get_relative_tolerance(figure_size) * 10

        for x, y in product(range(side_length), range(side_length)):
            center_distance = ImageGenerator._distance((x, y), center)
            if center_distance <= figure_size and (
                math.isclose(x, center[0], abs_tol=tolerance)
                or math.isclose(y, center[1], abs_tol=tolerance)
            ):
                image[x, y] = 1

        return image

    @staticmethod
    def _generate_random_figure(
        side_length: int, figure_size: int, centered: bool, noise: float
    ):
        """
        Generates a random figure.
        """
        center = ImageGenerator._calculate_figure_center(
            side_length, figure_size, centered
        )

        figure_generation_functions = [
            ImageGenerator._generate_circle,
            ImageGenerator._generate_cross,
        ]

        image = random.choice(figure_generation_functions)(
            side_length, figure_size, center
        )

        return ImageGenerator._add_noise(image, noise)

    @staticmethod
    def _conditional_flatten(image_set, flatten: bool):
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
        noise=0.003,
    ):
        """
        Generates images.
        """
        if round(sum(image_set_fractions), 5) != 1:
            raise ValueError("Image set fractions must sum to 1.")

        training, validation, test = image_set_fractions

        if not (0 <= training <= 1 and 0 <= validation <= 1 and 0 <= test <= 1):
            raise ValueError("All image set fractions must be numbers between 0 and 1.")

        image_set = [
            ImageGenerator._generate_random_figure(
                side_length=side_length,
                figure_size=random.randint(5, 50),
                centered=centered,
                noise=noise,
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