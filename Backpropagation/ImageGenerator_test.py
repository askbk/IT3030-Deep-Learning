from ImageGenerator import ImageGenerator


def test_image_generator_constructor():
    ImageGenerator()
    ImageGenerator(
        side_length=10,
        noise=0.1,
        figure_size_range=(7, 10),
        flatten=False,
    )


def test_generates_correct_number_of_images():
    training, validation, test = ImageGenerator().generate(image_set_size=50)
    assert len(training) + len(validation) + len(test) == 50


def test_generates_correct_set_fractions():
    training, validation, test = ImageGenerator().generate(
        image_set_fractions=(0.7, 0.2, 0.1), image_set_size=400
    )

    total = len(training) + len(validation) + len(test)

    assert len(training) / total == 0.7
    assert len(validation) / total == 0.2
    assert len(test) / total == 0.1


def test_correct_side_length():
    side_length = 10
    training, validation, test = ImageGenerator().generate(
        side_length=side_length, flatten=False
    )

    assert all(
        map(
            lambda image: len(image) == side_length and len(image[0]) == side_length,
            [*training, *validation, *test],
        )
    )


def test_flatten():
    side_length = 15
    training, validation, test = ImageGenerator().generate(
        flatten=True, side_length=side_length
    )
    print(test)
    assert all(
        map(
            lambda image: len(image) == side_length * side_length,
            [*training, *test, *validation],
        )
    )
