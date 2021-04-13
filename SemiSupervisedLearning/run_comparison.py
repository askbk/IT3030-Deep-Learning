from Classifier import Classifier
from Utils import get_preprocessed_data
from Visualization import graph_training_history


def comparison():
    (x_train_unlabeled, x_train_labeled, y_train_labeled), (
        x_test_unlabeled,
        x_test_labeled,
        y_test_labeled,
    ) = get_preprocessed_data("mnist", semi_supervised_split=True, labeled_fraction=0.5)

    supervised_history = Classifier.train_supervised(
        {
            "epochs": 3,
            "batch_size": 32,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
        },
        training_set=(x_train_labeled[:1000], y_train_labeled[:1000]),
        return_learning_progress=True,
    )
    semisupervised_history = Classifier.train_semisupervised(
        {
            "epochs": 1,
            "batch_size": 256,
            "optimizer": "adam",
            "loss": "mean_squared_error",
        },
        {
            "epochs": 3,
            "batch_size": 32,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
        },
        unlabeled_training_set=x_train_unlabeled[:1000],
        labeled_training_set=(x_train_labeled[:1000], y_train_labeled[:1000]),
        return_learning_progress=True,
    )
    graph_training_history(
        [
            ("supervised", supervised_history),
            ("semisupervised", semisupervised_history),
        ],
        keys=["accuracy"],
    )


if __name__ == "__main__":
    comparison()
