from matplotlib import pyplot as plt
from itertools import chain
import numpy as np


class ImageViewer:
    """
    Class for viewing images.
    """

    @staticmethod
    def display(image):
        """
        Displays an image.
        """
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    @staticmethod
    def display_images(images):
        _fig, axs = plt.subplots(2, 5, figsize=(10, 10))
        for ax, image in zip(chain.from_iterable(axs), images):
            ax.imshow(image)
            ax.grid(False)
            ax.axis("off")

        plt.show()


if __name__ == "__main__":
    from ImageGenerator import ImageGenerator

    image_set = ImageGenerator().generate(
        image_set_size=10,
        dataset_split=(1,),
        centered=False,
        side_length=16,
        figure_size_range=(10, 15),
    )[0]
    images, labels = list(zip(*image_set))
    print("".join([f"{label}\n" for label in labels]))
    ImageViewer.display_images(images)
