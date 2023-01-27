import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def dmse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    if y_pred.any() <= 0:
        print("trøbbel")
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def dbinary_cross_entropy(y_true, y_pred):
    return ((1-y_true) / (1-y_pred) - y_true / y_pred) / np.size(y_true)

def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))

def dcross_entropy(y_true, y_pred):
    return -y_true / y_pred / np.size(y_true)