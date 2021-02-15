class LayerBase:
    """
    Neural network layer interface
    """

    def forward_pass(self, data):
        raise NotImplementedError

    def backward_pass(self, J_L_Z, Z, Y):
        raise NotImplementedError

    def update_weights(self, jacobians, learning_rate):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError