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
        image_set_fractions=(0.7, 0.2, 0.1)
    )

    total = len(training) + len(validation) + len(test)

    assert len(training) / total == 0.7
    assert len(validation) / total == 0.2
    assert len(test) / total == 0.1
