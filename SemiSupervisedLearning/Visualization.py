import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Sequence


def graph_training_history(
    histories: Sequence[tf.keras.callbacks.History], title: str = None
):
    plt.figure(1)
    for history in histories:
        # summarize history for accuracy
        # plt.subplot(211)
        # plt.plot(history.history["acc"])
        # plt.plot(history.history["val_acc"])
        # plt.title("model accuracy")
        # plt.ylabel("accuracy")
        # plt.xlabel("epoch")
        # plt.legend(["train", "test"], loc="upper left")
        # summarize history for loss
        # loss, val_loss = history.history["loss"], history.history["val_loss"]
        keys = history.history.keys()
        for key in keys:
            plt.plot(history.history[key])
        # plt.plot(loss)
        # plt.plot(val_loss)
        plt.title(title)
        # plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(keys, loc="upper left")
        plt.show()
