import math
import random
import numpy as np
from functools import reduce
from itertools import chain, product, accumulate, islice
from typing import Tuple, List, Callable, Union

Image = List[List[int]]
FlattenedImage = np.array
Dataset = List[Tuple[Union[Image, FlattenedImage], str]]


class ImageGenerator:
    """
    Static class for generating images.
    """

    @staticmethod
    def _calculate_figure_center(
        side_length: int, figure_size: int, centered: bool
    ) -> Tuple[float, float]:
        """
        Calculate the center coordinates for a figure.
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
    def _l2_distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> int:
        """
        Calculate L2 distance between two points in 2D.
        """
        x1, y1 = point_a
        x2, y2 = point_b

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def _add_noise(image: Image, noise: float) -> Image:
        """
        Randomly flip pixels to add noise.
        """
        return [
            [pixel if random.random() > noise else 1 - pixel for pixel in row]
            for row in image
        ]

    @staticmethod
    def _generate_generic_figure(
        side_length: int, figure_function: Callable[[int, int], bool]
    ) -> Image:
        """
        Colors a pixel if figure_function(x, y) is true.
        """
        return [
            [int(figure_function(x, y)) for y in range(side_length)]
            for x in range(side_length)
        ]

    @staticmethod
    def _generate_circle(
        side_length: int, figure_size: int, center: Tuple[float]
    ) -> Tuple[Image, str]:
        """
        Generate a circle.
        """

        def should_be_colored(x, y):
            return math.isclose(
                figure_size / 2,
                ImageGenerator._l2_distance((x, y), center),
                abs_tol=0.8,
            )

        return (
            ImageGenerator._generate_generic_figure(
                side_length, figure_function=should_be_colored
            ),
            "circle",
        )

    @staticmethod
    def _generate_cross(
        side_length: int, figure_size: int, center: Tuple[float, float]
    ) -> Tuple[Image, str]:
        """
        Generates a cross
        """
        tolerance = 0.5

        def should_be_colored(x, y):
            return ImageGenerator._l2_distance((x, y), center) <= figure_size / 2 and (
                math.isclose(x, center[0], abs_tol=tolerance)
                or math.isclose(y, center[1], abs_tol=tolerance)
            )

        return (
            ImageGenerator._generate_generic_figure(
                side_length, figure_function=should_be_colored
            ),
            "cross",
        )

    @staticmethod
    def _generate_rectangle(
        side_length: int, figure_size: int, center: Tuple[float, float]
    ) -> Tuple[Image, str]:
        """
        Generates a rectangle
        """
        tolerance = 0.7

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

        return (
            ImageGenerator._generate_generic_figure(
                side_length, figure_function=should_be_colored
            ),
            "rectangle",
        )

    @staticmethod
    def _generate_triangle(
        side_length: int, figure_size: int, center: Tuple[float, float]
    ) -> Tuple[Image, str]:
        """
        Generates a triangle
        """
        leg_tolerance = 1.3
        base_tolerance = 0.6

        half_figure_side_lengths = [
            figure_size // 2,
            random.randint(figure_size // 2.5, figure_size // 2),
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

        return (
            ImageGenerator._generate_generic_figure(
                side_length, figure_function=should_be_colored
            ),
            "triangle",
        )

    @staticmethod
    def _generate_random_figure(
        side_length: int, figure_size: int, centered: bool, noise: float
    ) -> Tuple[Image, str]:
        """
        Generates a random figure.
        """
        image, figure_class = random.choice(
            [
                ImageGenerator._generate_circle,
                ImageGenerator._generate_cross,
                ImageGenerator._generate_rectangle,
                ImageGenerator._generate_triangle,
            ]
        )(
            side_length=side_length,
            figure_size=figure_size,
            center=ImageGenerator._calculate_figure_center(
                side_length, figure_size, centered
            ),
        )

        return (
            ImageGenerator._add_noise(
                image=image,
                noise=noise,
            ),
            figure_class,
        )

    @staticmethod
    def _conditional_flatten(
        image_set: List[Tuple[Image, str]], flatten: bool
    ) -> Dataset:
        """
        Flattens the image set if flatten is True.
        """
        if not flatten:
            return image_set

        return [
            (np.array(list(chain.from_iterable(image[0]))), image[1])
            for image in image_set
        ]

    @staticmethod
    def _split_image_set(
        image_set: Dataset, image_set_fractions: Tuple[float, ...]
    ) -> Tuple[Dataset, ...]:
        """
        Splits an image set according to given subset proportions.
        """
        if round(sum(image_set_fractions), 5) != 1:
            raise ValueError("Image set fractions must sum to 1.")

        if not all(map(lambda x: 0 < x <= 1, image_set_fractions)):
            raise ValueError("All image set fractions must be numbers between 0 and 1.")

        split_sizes = map(
            lambda fraction: int(fraction * len(image_set)), image_set_fractions
        )

        return (list(islice(image_set, 0, split_size)) for split_size in split_sizes)

    @staticmethod
    def _generate_random_figures(
        side_length: int,
        figure_size_range: Tuple[int],
        centered: bool,
        noise: float,
        image_set_size: int,
    ) -> List[Image]:
        """
        Generates a given number of random images.
        """
        return [
            ImageGenerator._generate_random_figure(
                side_length=side_length,
                figure_size=random.randint(figure_size_range[0], figure_size_range[1]),
                centered=centered,
                noise=noise,
            )
            for i in range(image_set_size)
        ]

    @staticmethod
    def generate(
        image_set_size=100,
        dataset_split=(0.7, 0.2, 0.1),
        side_length=10,
        figure_size_range=(5, 10),
        flatten=False,
        centered=False,
        noise=0.003,
    ) -> Tuple[Dataset, ...]:
        """
        Generates random images of circles, crosses, rectangles and triangles.
        """
        return tuple(
            ImageGenerator._conditional_flatten(image_set, flatten)
            for image_set in ImageGenerator._split_image_set(
                image_set=ImageGenerator._generate_random_figures(
                    side_length=side_length,
                    figure_size_range=figure_size_range,
                    centered=centered,
                    noise=noise,
                    image_set_size=image_set_size,
                ),
                image_set_fractions=dataset_split,
            )
        )


if __name__ == "__main__":
    from ImageViewer import ImageViewer

    dataset, *_ = ImageGenerator().generate(
        image_set_size=100, side_length=50, centered=False, figure_size_range=(15, 45)
    )
    images = [data[0] for data in dataset]
    ImageViewer.display_images(images)