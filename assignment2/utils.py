import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mse_plot(predictions, train_mean, train_std):
    """
    Args:
        prediction: list of (predictions, targets) pairs of forecast size + 1 (25)
        train_mean: dictionary containing mean of every feature used in the model as well as the target
        train_std: dictionary containing std of every feature used in the model as well as the target
    """
    MSEs = []

    # Calculate MSE over predictions and targets
    for p, t in predictions:
        p = np.array(p) * train_std['next_timestep_consumption'] + train_mean['next_timestep_consumption']
        t = t.numpy() * train_std['next_timestep_consumption'] + train_mean['next_timestep_consumption']
        squared_error = ((t - p) ** 2).tolist()
        MSEs.append(squared_error)

    mse_df = pd.DataFrame(np.array(MSEs[:-1]))

    # MSE plot
    mean = mse_df.mean().values
    std = mse_df.std().values
    x = range(len(mean))

    plt.figure(figsize=(8, 4), dpi=150)
    plt.title("Mean Squared Error")
    plt.plot(x, mean)
    plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.2)
    plt.show()

def prediction_target_plot(predictions, train_mean, train_std, index):
    """
    Args:
        prediction: list of (predictions, targets) pairs of forecast size + 1 (25)
        train_mean: dictionary containing mean of every feature used in the model as well as the target
        train_std: dictionary containing std of every feature used in the model as well as the target
        index: index of the test interval to display
    """
    p, t0 = predictions[index]

    p = np.array(p) * train_std['next_timestep_consumption'] + train_mean['next_timestep_consumption']      
    t0 = t0.numpy() * train_std['next_timestep_consumption'] + train_mean['next_timestep_consumption']      

    plt.figure(figsize=(8, 4), dpi=150)
    plt.xticks(range(len(p)))
    plt.plot(range(len(p)), p, label="Prediction")
    plt.plot(range(len(p)), t0, label="Target")
    plt.scatter(range(1, len(p)), p[1:], color="blue")
    plt.scatter(range(1, len(p)), t0[1:], color="orange")
    plt.legend()
    plt.show()
    
def generate_lstm_sequences(df, tw, pw):    
    data = dict()
    L = len(df)
    for i in range(L-tw):
        sequence = df[i:i+tw].values 
        target = df[i+tw:i+tw+pw].values

        data[i] = {'sequence': sequence, 'target': target}

    return data