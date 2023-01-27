from layer import Layer

class Activation(Layer):
    def __init__(self, activation_function, d_activation):
        self.activation = activation_function
        self.d_activation = d_activation

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_grad, lr):
        return self.d_activation(self.input) * output_grad