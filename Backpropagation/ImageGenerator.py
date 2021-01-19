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
        self, image_set_size=100, image_set_fractions=(0.7, 0.2, 0.1), side_length=10
    ):
        """
        Generates images.
        """
        training, validation, test = image_set_fractions
        return (
            [[[0] * side_length] * side_length] * int(image_set_size * training),
            [[[0] * side_length] * side_length] * int(image_set_size * validation),
            [[[0] * side_length] * side_length] * int(image_set_size * test),
        )
