import torch.nn as nn

def init_conv(channel_list):
    layers = []
    for i in range(len(channel_list) -1):
        layers.append(nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size = 3, stride = 2))
        layers.append(nn.BatchNorm2d(channel_list[i+1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Conv2d(channel_list[-1], 1, kernel_size = 1, stride = 1))
    return layers

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        channel_list = [3,16,32,64]
        self.conv = nn.Sequential(*init_conv(channel_list))
        self.linear = nn.Linear(475, 4)

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.shape[0]
        return self.linear(x.view(batch_size,-1)) 
