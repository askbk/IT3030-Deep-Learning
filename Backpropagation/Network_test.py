from Network import Network


def test_network_constructor():
    Network(
        layers=[], loss_function="mse", regularization=None, regularization_rate=0.001
    )
