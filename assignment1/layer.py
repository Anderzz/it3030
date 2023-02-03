class Layer:
    # base class that is inherited by all layers
    def __init__(self):
        self.input = None
        self.output = None
        self.grad = None

    def forward(self, X):
        pass

    def backward(self, output_grad, lr):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass