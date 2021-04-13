import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Sequence, Tuple


def graph_training_history(
    histories: Sequence[Tuple[str, tf.keras.callbacks.History]],
    keys: Sequence[str],
    title: str = None,
):
    plt.figure(1)
    keys_to_use = list(
        set(
            filter(
                lambda key: any(filter_key in key for filter_key in keys),
                *(history.history.keys() for _, history in histories),
            )
        )
    )
    legend_keys = []
    for name, history in histories:
        for key in history.history.keys():
            if key in keys_to_use:
                plt.plot(history.history[key])
                legend_keys.append(f"{name} {key}")
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend(legend_keys, loc="upper left")
    plt.show()
