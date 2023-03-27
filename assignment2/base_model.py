import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
