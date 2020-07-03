import gym
import random
from math import exp
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from helper import decompose_sample
from model import DQN
from replaymemory import Replay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preproc(image):
    image = image.astype("float32")
    image = (image / 127.5) - 1
    image = image.transpose(-1,0,1)
    return image

def main(args):
    memory = Replay(args.num_memory)
    loss_f = nn.SmoothL1Loss()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())

    def optimize_model():
        if len(memory) < args.batch_size:
            return
        samples = memory.sample(args.batch_size)
        states, actions, rewards, boot_strap_estimates = decompose_sample(target_net,samples, device)
        q_s_a = policy_net(states).gather(1,actions)
        optimizer.zero_grad()
        loss = loss_f(q_s_a, args.gamma * boot_strap_estimates + rewards)
        # print("Huber Loss : {}".format(loss.item()))
        loss.backward()
        optimizer.step()

    def select_action(state,t):
        sample = random.random()
        eps_thresh = args.eps_start * exp(-1 * t / args.eps_decay)
        if sample > eps_thresh:
            state = torch.tensor(state).unsqueeze(0).float().to(device)
            with torch.no_grad():
                return policy_net(state).max(1)[1].item()
        else:
            return random.randint(0,3)
        
    env = gym.make('Breakout-v0')
    for e in tqdm(range(args.num_epoch)):
        if args.render: 
            env.render()
        state = env.reset()
        state = preproc(state)
        next_state = None
        done = False
        rewards = []
        max_steps = 200
        while not done:
            action = select_action(state, e)
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                next_state = None
            else:
                next_state = preproc(next_obs) - state
            memory.push(state, action, reward, next_state)
            state = next_state
            max_steps -= 1
            if max_steps == 0:
                break
        optimize_model()
        print("Total reward in the episode : {}, epoch : {}".format(sum(rewards), e+1))
        if e % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
        'batch_size' : 500,
        'num_epoch' : 300,
        'target_update' : 10,
        'gamma': 0.999,
        'num_memory': 10000,
        'eps_start': 0.9,
        'eps_decay': 200,
        'render': False
    }
    args.update(args_dict)
    main(args)
