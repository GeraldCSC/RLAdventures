import numpy as np
import gym
import torch 
from memory import ReplayBuffer
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn as nn
from model import DQNLinear
import random


class TrainConfig:
    num_epochs = 10000
    batch_size = 10000
    lr = 0.00001
    gamma = 0.999
    capacity = 10000
    eps_start = 1
    eps_end = 0.1
    render = True
    monitor = True
    vid_interval = 1000
    vid_save_path = "videos/"
    save_path = "models/model.pt"
    grad_norm_clip = 1
    target_update = 10

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, policyclass, config):
        self.config = config
        self.env = gym.make(config.env)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.a_space, self.obs_space = self.env.action_space.n, self.env.observation_space.shape[0]
        self.policy_net = policyclass(self.obs_space, self.a_space).to(self.device)
        self.target_net = policyclass(self.obs_space, self.a_space).to(self.device)
        self.target_net.eval()
        self.buf = ReplayBuffer(config.capacity)
        self.lossfn = nn.MSELoss()
        self.optimizer = AdamW(self.policy_net.parameters(), lr = config.lr)
        self.eps = config.eps_start
        self.eps_interval = (config.eps_start - config.eps_end) /config.num_epochs 
        self.eps_interval *= 2
        if config.render:
            self.env.render()
        if config.monitor:
            self.env = gym.wrappers.Monitor(self.env, config.vid_save_path, \
                                            video_callable = lambda ep: ep % config.vid_interval == 0,force= True)

    def zero_grad(self):
        for param in self.policy_net.parameters():
            param.grad = None

    def train(self):
        config, env, buf = self.config, self.env, self.buf
        lr = config.lr

        def update_target_net():
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def run_epoch():
            loss = None
            curr_state = env.reset()
            done, next_state, reward_list = False, None, []
            while not done:
                action = self.get_eps_act(torch.tensor(curr_state, device = self.device, dtype = torch.float32).unsqueeze(0))
                next_state, reward, done, _ = env.step(action)
                reward_list.append(reward)
                self.buf.push(curr_state, action, reward, next_state, done)
                curr_state = next_state
                if len(self.buf) >= self.config.batch_size:
                    loss = self.optimize_model()
            return sum(reward_list), loss
                
        pbar = tqdm(range(config.num_epochs))
        for eps in pbar:
            rewards, loss = run_epoch()
            if eps % config.target_update == 0:
                update_target_net()
            if loss is not None:
                strprint = f"epoch {eps+1}: loss {loss:.5f}. eps {self.eps} reward {rewards}"
            else:
                strprint = f"epoch {eps+1}: eps {self.eps} reward {rewards}"
            pbar.set_description(strprint)
            if self.eps > self.config.eps_end:
                self.eps = self.eps - self.eps_interval
        self.save_model()

    def optimize_model(self):
        S, A, R, S_, done = self.buf.torch_samples(self.config.batch_size, device =self.device)
        target = self.config.gamma * self.target_net(S_).max(1)[0].detach().view(-1,1) * (1-done)
        target = target + R
        estimate = self.policy_net(S).gather(1, A)
        loss = self.lossfn(estimate, target)
        self.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config.grad_norm_clip)
        self.optimizer.step()
        return loss.item()

    def get_eps_act(self, state):
        """
            accepts a tensor that is loaded onto the device already
        """
        if random.random() > self.eps:
            action = self.policy_net(state).max(1)[1].item()
        else:
            action = random.randrange(self.a_space)
        return action

    def save_model(self):
        logger.info("Saving Model to {self.config.save_path}")
        torch.save(self.policy_net.state_dict(), self.config.save_path)

def run(**kwargs):
    config = TrainConfig(**kwargs)
    trainer = Trainer(DQNLinear, config)
    trainer.train()

if __name__ == "__main__":
    run(num_epochs=8000,lr=0.0001, capacity = 1000,batch_size = 256,env = "CartPole-v0", render = False)
