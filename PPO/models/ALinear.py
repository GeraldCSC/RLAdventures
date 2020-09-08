import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class LinearAgent(nn.Module):
    def __init__(self, input_size = 128, hidden_size = 256, action_space = 4):
        super(LinearAgent, self).__init__() 
        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(inplace = True), \
                                   nn.Linear(hidden_size, action_space))
    def forward(self, x):
        raise NotImplementedError

    def get_dist(self,x):
        logits = self.actor(x)
        dist = Categorical(logits = logits)
        return dist

    def get_action(self,x):
        dist = self.get_dist(x)
        return dist.sample().item()
