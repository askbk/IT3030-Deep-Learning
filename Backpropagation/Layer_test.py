from Layer import Layer


def test_layer_constructor():
    Layer()
    Layer(
        neurons=5,
        activation_function="sigmoid",
        softmax=False,
        initial_weight_range=(-0.1, 0.1),
    )
