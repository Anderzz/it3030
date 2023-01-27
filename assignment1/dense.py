from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size, regularization = '', reg_lambda = 0.0):
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        # do xavier initialization of the weights
        self.w = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.b = np.random.randn(output_size, 1) / np.sqrt(input_size)


    def forward(self, input):
        self.input = input
        return np.einsum('ij,jk->ik', self.w, self.input) + self.b
        

    def backward(self, output_gradient, lr):
        biases_gradient =  np.zeros((output_gradient.shape[0], 1)) # needed incase of no regularization
        weights_gradient = np.einsum('ij,ik->ik', output_gradient, self.input.T)
        input_gradient = np.einsum('ij,jk->ik', self.w.T, output_gradient)
        if self.regularization == "l1":
            weights_gradient += self.reg_lambda * np.sign(self.w)
            biases_gradient = self.reg_lambda * np.sign(self.b)
        elif self.regularization == "l2":
            weights_gradient += self.reg_lambda * self.w
            biases_gradient = self.reg_lambda * self.b

        # update weights and biases
        self.w -= lr * weights_gradient
        self.b -= lr * (output_gradient + biases_gradient)

        return input_gradient

    def get_weights(self):
        return self.w, self.b

    def set_weights(self, weights):
        self.w = weights[0]
        self.b = weights[1]