import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Model(nn.Module):
    # create a lstm model
    def __init__(self, input_size=10, hidden_size=100, num_layers=1, output_size=1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    


# Define hyperparameters
input_size = 10
hidden_size = 64
output_size = 1
num_layers = 1
sequence_length = 24
learning_rate = 0.01
num_epochs = 100

# Create LSTM model instance
model = Model(input_size, hidden_size, output_size, num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# load data
train_data = pd.read_csv('data/train.csv')

