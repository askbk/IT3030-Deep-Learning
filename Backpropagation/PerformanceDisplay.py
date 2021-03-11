import matplotlib.pyplot as plt
import numpy as np


class PerformanceDisplay:
    @staticmethod
    def display_performance(training, validation, test):
        training_length = len(training)
        test_stretched = np.full(int(0.05 * training_length), test)
        x_training = np.arange(training_length)
        x_test = np.arange(training_length, int(training_length * 1.05))

        plt.plot(x_training, training, label="training")
        if validation:
            plt.plot(x_training, validation, label="validation")
        plt.plot(x_test, test_stretched, label="test")

        plt.legend()
        plt.draw()
        plt.show()
