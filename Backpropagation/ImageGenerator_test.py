from ImageGenerator import ImageGenerator


def test_image_generator_constructor():
    ImageGenerator()
    ImageGenerator(
        side_length=10,
        noise=0.1,
        image_set_fractions=(0.7, 0.2, 0.1),
        figure_size_range=(7, 10),
        flatten=False,
        image_set_size=100,
    )
