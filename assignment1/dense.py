from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size, regularization = '', reg_lambda = 0.0, lr = 0.0, wr = None, br = None):
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.lr = lr
        # if a weight range is not specified, use glorot initialization
        if wr is None:
            self.w = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        else:
            self.w = np.random.uniform(wr[0], wr[1], (output_size, input_size))

        # if a bias range is not specified, use glorot initialization
        if br is None:
            self.b = np.zeros((1, output_size))
        else:
            self.b = np.random.uniform(br[0], br[1], (output_size, 1))

    def forward(self, input):
        self.input = input
        return np.einsum('ij,jk->ik', self.w, self.input) + self.b
        

    def backward(self, output_gradient, lr):
        biases_gradient =  np.zeros((output_gradient.shape[0], 1)) # needed incase of no regularization
        weights_gradient = np.einsum('ij,ik->ik', output_gradient, self.input.T)
        input_gradient = np.einsum('ij,jk->ik', self.w.T, output_gradient)
        # if lr is not specified, use the global one passed in from train
        if self.lr == 0.0:
            self.lr = lr
        if self.regularization == "l1":
            weights_gradient += self.reg_lambda * np.sign(self.w)
            biases_gradient = self.reg_lambda * np.sign(self.b)
        elif self.regularization == "l2":
            weights_gradient += self.reg_lambda * self.w
            biases_gradient = self.reg_lambda * self.b

        # update weights and biases
        self.w -= self.lr * weights_gradient
        self.b -= self.lr * (output_gradient + biases_gradient)

        return input_gradient

    def get_weights(self):
        return self.w, self.b

    def set_weights(self, weights):
        self.w = weights[0]
        self.b = weights[1]