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

    images = ImageGenerator().generate()[0][:10]
    ImageViewer.display_images([np.random.rand(50, 50) for i in range(10)])
    ImageViewer.display_images(images)
