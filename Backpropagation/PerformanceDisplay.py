import matplotlib.pyplot as plt
import numpy as np


class PerformanceDisplay:
    @staticmethod
    def display_performance(training, validation, test):
        training_length = len(training)
        test_length = int(0.1 * training_length)
        test_stretched = np.full(test_length, test)
        x_training = np.arange(training_length)
        x_test = np.arange(training_length, training_length + test_length)

        plt.plot(x_training, training, label="training")
        if validation is not None:
            plt.plot(x_training, validation, label="validation")
        plt.plot(x_test, test_stretched, label="test")

        plt.legend()
        plt.draw()
        plt.show()
