import numpy as np

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