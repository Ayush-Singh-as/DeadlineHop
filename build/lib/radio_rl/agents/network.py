from torch import nn


class QNetwork(nn.Module):
    def __init__(self, history=16, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(48*history, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 9))

    def forward(self, x):
        return self.layers(x)

