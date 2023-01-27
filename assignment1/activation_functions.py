import numpy as np
from activation import Activation
from layer import Layer

class Tanh(Activation):
    def __init__(self):

        def tanh(x):
            return np.tanh(x)

        def dtanh(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, dtanh)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def dsigmoid(x):
            z = sigmoid(x)
            return z * (1 - z)

        super().__init__(sigmoid, dsigmoid)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_grad, lr):
        n = np.size(self.output)
        return np.dot((np.eye(n) - self.output.T) * self.output, output_grad)

class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(x, 0)

        def drelu(x):
            return 1 * (x > 0)

        super().__init__(relu, drelu)

class LeakyRelu(Activation):
    def __init__(self, alpha = 0.01):
        def leaky_relu(x):
            return np.maximum(x, alpha * x)

        def dleaky_relu(x):
            return 1 * (x > 0) + alpha * (x < 0)

        super().__init__(leaky_relu, dleaky_relu)

class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def dlinear(x):
            return 1

        super().__init__(linear, dlinear)