import numpy as np
import matplotlib.pyplot as plt

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train_val_split(x, y, val_ratio = 0.2):
    val_size = int(len(x) * val_ratio)
    x_train = x[val_size:]
    y_train = y[val_size:]
    x_val = x[:val_size]
    y_val = y[:val_size]
    return x_train, y_train, x_val, y_val

def train(network, loss, dloss, x_train, y_train, x_val = None, y_val = None, epochs = 1000, lr = 0.01, verbose = True, batch_size = 1, early_stopping=True, patience=10):
    """ Train a neural network using mini-batch gradient descent with early stopping and data shuffling.

    Args:
        network (_type_): Neural network
        loss (_type_): Loss function
        dloss (_type_): Derivative of loss function
        x_train (_type_): Training data
        y_train (_type_): Training labels
        epochs (int, optional): Defaults to 1000.
        lr (float, optional): Global learning rate. Defaults to 0.01.
        verbose (bool, optional): Defaults to True.
        batch_size (int, optional): Size of mini-batch. Defaults to 1.
        early_stopping (bool, optional): Defaults to True.
        patience (int, optional): How many steps where validation loss does not improve before early stopping. Defaults to 10.
    """
    # if we don't have validation data, split training data into training and validation
    if not x_val:
        x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
    errors = []
    val_errors = []
    curr_best_val_error = np.inf
    best_weights = None
    counter = 0

    for epoch in range(epochs):
        epoch_error = 0
        total_batch = int(len(x_train) / batch_size)

        # shuffle training data
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]
        for i in range(total_batch):
            error = 0
            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]
            for x, y in zip(x_batch, y_batch):
                # forward pass
                output = predict(network, x)
                error += loss(y, output)
                # backward pass
                grad = dloss(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, lr)

        # store errors for plotting
            error /= len(x_train)
            epoch_error += error
        epoch_error /= total_batch
        errors.append(epoch_error)
        val_error = 0

        # compute validation error
        for x, y in zip(x_val, y_val):
            val_output = predict(network, x)
            val_error += loss(y, val_output)
        val_error /= len(x_val)
        val_errors.append(val_error)
        
        if verbose:
            print(f"epoch {epoch+1}/{epochs} error = {epoch_error} val_error = {val_error}")
        
        # early stopping
        if val_error < curr_best_val_error:
            curr_best_val_error = val_error
            best_weights = [layer.get_weights() for layer in network]
        else:
            counter += 1
            if early_stopping and counter >= patience:
                if verbose:
                    print(f"No improvement in validation loss for {patience} epochs! Early stopping after {epoch+1} epochs.")
                break
    # update the layers with the weights with the lowest validation error
    for i, layer in enumerate(network):
        layer.set_weights(best_weights[i])

    if verbose:
        plt.plot(errors)
        plt.plot(val_errors)
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend(['train', 'val'])
        plt.show()