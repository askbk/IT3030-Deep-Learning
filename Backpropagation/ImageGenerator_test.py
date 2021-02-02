from ImageGenerator import ImageGenerator
import pytest
from itertools import chain


def test_generates_correct_number_of_images():
    training, validation, test = ImageGenerator.generate(image_set_size=50)
    assert len(training) + len(validation) + len(test) == 50


def test_generates_correct_set_fractions():
    training, validation, test = ImageGenerator.generate(
        image_set_fractions=(0.7, 0.2, 0.1), image_set_size=400
    )

    total = len(training) + len(validation) + len(test)

    assert len(training) / total == 0.7
    assert len(validation) / total == 0.2
    assert len(test) / total == 0.1


def test_correct_side_length():
    side_length = 10
    training, validation, test = ImageGenerator.generate(
        side_length=side_length, flatten=False
    )

    assert all(
        map(
            lambda image: len(image[0]) == side_length
            and len(image[0][0]) == side_length,
            [*training, *validation, *test],
        )
    )


def test_flatten():
    side_length = 15
    training, validation, test = ImageGenerator.generate(
        flatten=True, side_length=side_length
    )

    assert all(
        map(
            lambda image: len(image[0]) == side_length * side_length,
            [*training, *test, *validation],
        )
    )


def test_invalid_image_set_fraction_throws_exception():
    with pytest.raises(ValueError):
        ImageGenerator.generate(image_set_fractions=(-1, 2, 0))

    with pytest.raises(ValueError):
        ImageGenerator.generate(image_set_fractions=(0.6, 0.6, 0.1))

    with pytest.raises(ValueError):
        ImageGenerator.generate(image_set_fractions=(0.2, 0.2, 0.2))


def test_images_are_binary_arrays():
    training, validation, test = ImageGenerator.generate(centered=True)

    assert all(
        map(
            lambda image: all(
                map(
                    lambda pixel: pixel == 1 or pixel == 0,
                    chain.from_iterable(image[0]),
                )
            ),
            [*training, *validation, *test],
        )
    )