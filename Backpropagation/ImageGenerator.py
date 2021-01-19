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
    def _generate_generic_figure(side_length, figure_function):
        """
        Colors a pixel if figure_function(x, y) is true.
        """

        return list(
            map(
                lambda x: list(
                    map(
                        lambda y: 1 if figure_function(x, y) else 0,
                        range(side_length),
                    )
                ),
                range(side_length),
            ),
        )

    @staticmethod
    def _generate_circle(side_length: int, figure_size: int, center: Tuple[float]):
        """
        Generates a circle
        """
        tolerance = 0.3 / math.log2(figure_size)

        def should_be_colored(x, y):
            center_distance = ImageGenerator._distance((x, y), center)
            return math.isclose(
                figure_size / 2,
                center_distance,
                rel_tol=tolerance,
            )

        return ImageGenerator._generate_generic_figure(
            side_length, figure_function=should_be_colored
        )

    @staticmethod
    def _generate_cross(side_length: int, figure_size: int, center: Tuple[float]):
        """
        Generates a cross
        """
        tolerance = math.log2(figure_size) * 0.3

        def should_be_colored(x, y):
            center_distance = ImageGenerator._distance((x, y), center)
            return center_distance <= figure_size and (
                math.isclose(x, center[0], abs_tol=tolerance)
                or math.isclose(y, center[1], abs_tol=tolerance)
            )

        return ImageGenerator._generate_generic_figure(
            side_length, figure_function=should_be_colored
        )

    @staticmethod
    def _generate_rectangle(side_length: int, figure_size: int, center: Tuple[float]):
        """
        Generates a rectangle
        """
        tolerance = 1

        half_figure_side_lengths = [
            figure_size // 2,
            random.randint(figure_size // 4, figure_size // 2),
        ]

        v_deviation, h_deviation = random.sample(half_figure_side_lengths, k=2)

        def lte(a, b):
            return a <= b or math.isclose(a, b, abs_tol=tolerance)

        def should_be_colored(x, y):
            v_distance = abs(y - center[1])
            h_distance = abs(x - center[0])
            return (
                math.isclose(v_distance, v_deviation, abs_tol=tolerance)
                and lte(h_distance, h_deviation)
            ) or (
                math.isclose(h_distance, h_deviation, abs_tol=tolerance)
                and lte(v_distance, v_deviation)
            )

        return ImageGenerator._generate_generic_figure(
            side_length, figure_function=should_be_colored
        )

    @staticmethod
    def _generate_triangle(side_length: int, figure_size: int, center: Tuple[float]):
        """
        Generates a triangle
        """
        # leg_tolerance = 5 / math.log2(figure_size)
        leg_tolerance = math.sqrt(figure_size * 1.1) * 0.4
        base_tolerance = math.sqrt(figure_size * 0.5) * 0.3

        half_figure_side_lengths = [
            figure_size // 2,
            random.randint(figure_size // 3, figure_size // 2),
        ]

        def get_slope(point_a, point_b):
            return (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])

        def get_constant(point, slope):
            return point[1] - slope * point[0]

        v_deviation, h_deviation = random.sample(half_figure_side_lengths, k=2)
        top = (center[0], center[1] + v_deviation)
        left = (center[0] - h_deviation, center[1] - v_deviation)
        right = (center[0] + h_deviation, center[1] - v_deviation)
        left_slope = get_slope(left, top)
        left_constant = get_constant(left, left_slope)
        right_slope = get_slope(top, right)
        right_constant = get_constant(top, right_slope)

        def is_on_left_leg(x, y):
            return (
                math.isclose(y, left_slope * x + left_constant, abs_tol=leg_tolerance)
                and left[0] <= x <= top[0]
            )

        def is_on_right_leg(x, y):
            return (
                math.isclose(y, right_slope * x + right_constant, abs_tol=leg_tolerance)
                and top[0] <= x <= right[0]
            )

        def is_on_base(x, y):
            return (
                math.isclose(y, center[1] - v_deviation, abs_tol=base_tolerance)
                and left[0] <= x <= right[0]
            )

        def should_be_colored(x, y):
            return is_on_left_leg(x, y) or is_on_right_leg(x, y) or is_on_base(x, y)

        return ImageGenerator._generate_generic_figure(
            side_length, figure_function=should_be_colored
        )

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
            ImageGenerator._generate_rectangle,
            ImageGenerator._generate_triangle,
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