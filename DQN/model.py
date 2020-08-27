import torch.nn as nn

class DQNLinear(nn.Module):
    def __init__(self, inputsize = 128, action_space = 4):
        super().__init__()
        x = [nn.Linear(inputsize, 256), nn.ReLU()]
        x += [nn.Linear(256, 128), nn.ReLU()]
        x += [nn.Linear(128, 64), nn.ReLU()]
        x += [nn.Linear(64,action_space)]
        self.go = nn.Sequential(*x)

    def forward(self, x):
        return self.go(x)
