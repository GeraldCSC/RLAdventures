from model import DQN
from torch.optim import Adam
from memory import Replay
import random
from math import exp
import torch
from helper import decompose_sample, select_action

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_DECAY = 200
TARGET_UPDATE = 10

def optimize_model(loss_f, policy_net, target_net, optimizer, samples, batch_size, gamma,device):
    loss_f = nn.SmoothL1Loss()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())
    memory = Replay(300)
    states, actions, rewards, boot_strap_estimates = decompose_sample(target_net,memory.sample(batch_size), device)
    q_s_a = policy_net(states).gather(1,actions)
    optimizer.zero_grad()
    loss = loss_f(q_s_a, gamma * boot_strap_estimates + rewards)
    loss.backward()
    optimizer.step()
