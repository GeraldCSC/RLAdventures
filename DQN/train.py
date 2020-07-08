import gym
import numpy as np
import random
from math import exp
from tqdm import tqdm
import torch
import torch.nn as nn
from model import DQN
from memory import ReplayBuffer
from helper import preproc, get_boot_strap_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    replay_buffer = ReplayBuffer(args.num_memory)
    loss_f = nn.SmoothL1Loss()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters())

    def optimize_model():
        if len(replay_buffer) < args.batch_size:
            return
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        state = torch.tensor(state).float().to(device)
        action = torch.tensor(action).long().to(device)
        reward = torch.tensor(reward).float().to(device)
        next_state = torch.tensor(next_state).float().to(device)
        done = torch.tensor(done).to(device)
        boot_strap_value = get_boot_strap_value(target_net, next_state, done)
        optimizer.zero_grad()
        q_s_a = policy_net(state).gather(1,action)
        loss = loss_f(q_s_a, (args.gamma * boot_strap_value) + reward)
        loss.backward()
        optimizer.step()

    def select_action(state, e):
        """
            state must be preproced
        """
        if e != 0:
            eps_thresh = args.eps_thresh / e
        else:
            eps_thresh = args.eps_thresh
        sample = random.random()
        if sample > eps_thresh:
            state = torch.tensor(state).unsqueeze(0).to(device)
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
        count = 0
        while not done:
            action = select_action(state, e)
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                next_state = np.zeros_like(state)
            else:
                next_state = preproc(next_obs) - state
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            optimize_model()
        print("Total reward in the episode : {}, epoch : {}".format(sum(rewards), e+1))
        if (e % args.target_update) == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), "model.pt")

if __name__ == "__main__":
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
        'batch_size' : 300,
        'num_epoch' : 300,
        'target_update' : 10,
        'gamma': 0.999,
        'num_memory': 1000,
        'eps_thresh': 0.85,
        'render': False
    }
    args.update(args_dict)
    main(args)
