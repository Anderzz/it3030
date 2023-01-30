import numpy as np
from doodler import *
import matplotlib.pyplot as plt
from network import *
from dense import Dense
from activation import Activation
from loss import *
from activation_functions import *
from parse_network import parse_file


def train_test_split(im, targets, train_size=0.8):
    # split into train and test
    split = int(len(im) * train_size)
    x_train, x_test = im[:split], im[split:]
    y_train, y_test = targets[:split], targets[split:]
    return x_train, x_test, y_train, y_test

def preprocess(x):
    x = x.astype("float32") / 255
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], 1)
    return x

def one_hot(y, num_classes):
  y = np.eye(num_classes)[y][:, 0, :]
  return y.reshape(y.shape[0], num_classes, 1)

n = 28
im, targets, labels, dim, flat = gen_standard_cases(count=600, types=['ball', 'frame', 'triangle', 'polygon', 'flower'], rows=n, cols=n , noise=0, show=False)
x_train, x_test, y_train, y_test = train_test_split(im, targets, train_size=0.8)
x_train = preprocess(x_train)
x_test = preprocess(x_test)
y_train = one_hot(y_train, 5)
y_test = one_hot(y_test, 5)
out_dim = len(targets[0])


network, loss, dloss, lr, wlambda, wrt = parse_file('2_hidden.txt')

# network = [
#     Dense(n*n, 100),
#     Tanh(),
#     Dense(100, 50, regularization='l2', reg_lambda=0.001, lr=0.1),
#     Sigmoid(),
#     Dense(50, 12, regularization='l2', reg_lambda=0.001, lr=0.1),
#     Relu(),
#     Dense(12, out_dim, regularization='l2', reg_lambda=0.0, lr=0.4),
#     Softmax()
# ]

train(network, loss, dloss, x_train, y_train, epochs=100, lr=lr, batch_size = 4, patience=20, shuffle=True)

# test
accuracy = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    accuracy += np.argmax(output) == np.argmax(y)
accuracy /= len(x_test)
print('accuracy:', accuracy)