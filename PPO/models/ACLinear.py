import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)

class ACLinear(nn.Module):
    def __init__(self, input_size = 128, hidden_size = 256, action_space = 4):
        super(ACLinear, self).__init__() 
        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(), \
                                   nn.Linear(hidden_size, action_space))
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(), \
                                   nn.Linear(hidden_size, 1))
        self.apply(weight_init)

    def get_dist_value(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        dist = Categorical(logits = logits)
        return dist, value

    def forward(self, x):
        raise NotImplementedError

