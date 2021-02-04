import matplotlib.pyplot as plt
import numpy as np


class PerformanceDisplay:
    @staticmethod
    def display_performance(training, validation, test):
        total = len(training) + len(test)
        x = np.arange(total)
        plt.plot(x, training)
        plt.plot(x, validation)
        plt.draw()
        plt.show()