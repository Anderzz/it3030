import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def generate_sequence(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
    """Generate sequences for each target column

    Args:
        df (pd.DataFrame): Dataframe containing the data
        tw (int): Time window
        pw (int): Prediction window
        target_columns (list): List of target columns
        drop_targets (bool, optional): Drop targets from the dataframe. Defaults to False.

    Returns:
        list: List of sequences
    """    
    data = dict()
    L = len(df)
    for i in range(L-tw):
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)
        

        # get current sequence
        seq = df.iloc[i:i+tw].values

        # get current target
        target = df.iloc[i+tw:i+tw+pw][target_columns].values
        data[i] = {'seq': seq, 'target': target}
    return data

class SequenceDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.keys = list(data.keys())
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['seq']), torch.Tensor(sample['target'])
    

class Model(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda = False, dropout=0.2):
        """
        n_features: number of input features (1 is for univariate)
        n_hidden: number of hidden units in each layer
        n_outputs: number of outputs to predict for each training example
        sequence_len: how many time steps to look back
        n_lstm_layers: number of LSTM layers
        n_deep_layers: number of dense layers
        use_cuda: whether to use cuda
        dropout: dropout rate
        """
        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.n_hidden = n_hidden
        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_lstm_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(n_hidden*sequence_len, n_hidden)

        self.dropout = nn.Dropout(p=dropout)

        # Create fully connected layers (n_hidden x n_deep_layers)
        dnn_layers = []
        for i in range(n_deep_layers):
        # Last layer (n_hidden x n_outputs)
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(n_hidden, n_outputs))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(n_hidden, n_hidden))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))
        # compile DNN layers
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):

        # init hidden state
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.n_hidden)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.n_hidden)

        if self.use_cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        self.hidden = (hidden_state, cell_state)

        # forward
        x, h = self.lstm(x, self.hidden) # lstm
        x = self.dropout(x.contiguous().view(x.shape[0], -1)) # flatten
        x = self.fc1(x) # dense
        x = self.dnn(x) # dnn





def main():
    BATCH_SIZE = 32
    split_ratio = 0.8
    n_in, n_out = 24, 1

    n_hidden = 50
    n_dnn_layers = 5
    seq_len = 24
    n_inputs = 1

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    model = Model(n_inputs, n_hidden, n_out, seq_len, n_dnn_layers, USE_CUDA).to(device)

    df  = pd.read_csv('./data/simple_df.csv')
    sequences = generate_sequence(df, n_in, n_out, 'consumption')
    dataset = SequenceDataset(sequences)

    train_len = int(len(dataset)*split_ratio)
    lens = [train_len, len(dataset)-train_len]
    train_set, val_set = random_split(dataset, lens)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


if __name__ == '__main__':
    main()