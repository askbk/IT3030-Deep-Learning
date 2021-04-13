from Classifier import Classifier
from Autoencoder import Autoencoder
from Utils import get_preprocessed_data
from Visualization import graph_training_history, display, tsne_plot
import json


def comparison(config_path):
    with open(config_path) as f:
        config = json.loads(f.read())

    (x_train_unlabeled, x_train_labeled, y_train_labeled), (
        x_test_unlabeled,
        x_test_labeled,
        y_test_labeled,
    ) = get_preprocessed_data(
        "mnist",
        semi_supervised_split=True,
        labeled_fraction=config.get("labeled_fraction"),
    )

    autoencoder, auto_history = Autoencoder.train(
        config.get("autoencoder"),
        unlabeled=x_train_unlabeled,
        return_learning_progress=True,
    )

    graph_training_history([("autoencoder", auto_history)], keys=["loss"])
    predictions = autoencoder.predict(x_test_unlabeled[:100])
    display(x_test_unlabeled, predictions, n=config.get("reconstructions_to_display"))

    supervised_history = Classifier.train_supervised(
        config.get("supervised_classifier"),
        training_set=(x_train_labeled, y_train_labeled),
        return_learning_progress=True,
    )
    semisupervised_history = Classifier.train_semisupervised(
        config.get("autoencoder"),
        config.get("semisupervised_classifier"),
        unlabeled_training_set=x_train_unlabeled,
        labeled_training_set=(x_train_labeled, y_train_labeled),
        return_learning_progress=True,
        tsne_plot=config.get("tsne_plot", False),
    )
    graph_training_history(
        [
            ("supervised", supervised_history),
            ("semisupervised", semisupervised_history),
        ],
        keys=["accuracy"],
    )


if __name__ == "__main__":
    comparison("./config.json")
