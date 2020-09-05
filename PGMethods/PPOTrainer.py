from .trainerbase import TrainerConfig, Trainer
from collections import namedtuple
from ..utils.test import f
from torch.optim import Adam
from utils import zero_grad
from model import LinearPPOAgent
from memory import PGMemory
import torch

def ppo_update(model,optimizer,states, actions, log_probs, advantages, returns, num_epochs, eps_clip=0.2,entropy_coef = 0.01, value_w = 0.5):
    for epoch in range(num_epochs):
        dist, value = model(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions.flatten()).unsqueeze(1)
        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1- eps_clip, 1+ eps_clip) * advantages
        actor_loss = - torch.min(surr1, surr2).mean()
        critic_loss = (returns - value).pow(2).mean()
        loss = value_w * critic_loss + actor_loss - entropy_coef*entropy
        zero_grad(model)
        loss.backward()
        optimizer.step()

Tracker = namedtuple('Tracker', ['state', 'act', 'log_prob', 'value'])
class PPOTrainerConfig(TrainerConfig):
    max_frames = 1000000
    num_epochs = 30
    lr = 2.4e-4
    update_timestep = 128
    batch_size = 64
    render = False
    monitor = True
    st_advantage = True
    vid_interval = 1000
    vid_save_path = "videos/"
    save_path = "models/model.pt"
    env_id = "CartPole-v1"
    gamma = 0.99
    lam = 0.95
    eps_clip = 0.2 
    entropy_coef = 0.01
    value_w = 0.5
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PPOTrainer(Trainer): 
    def __init__(self, policyclass, config):
        super().__init__(config)
        self.curr_agent = policyclass(input_size=self.obs_space, action_space=self.a_space).to(self.device)
        self.optimizer = Adam(self.curr_agent.parameters(), lr = config.lr)
        self.tracker = None
        self.memory = PGMemory()
        self.counter = 0 #decides when to update

    def act(self, x):
        torch_obs = self.get_float_tensor(x)
        dist, value = self.curr_agent(torch_obs)
        action, value = dist.sample().detach(), value.item()
        log_prob = dist.log_prob(action).item()
        self.tracker = Tracker(x, action, log_prob, value)
        return action.item()

    def update(self, reward, next_state, done):
        t, config,self.counter = self.tracker, self.config ,self.counter + 1
        self.memory.push(t.state, t.act, t.log_prob, t.value, reward, done)
        self.tracker = None
        if self.counter == self.config.update_timestep:
            next_state = self.get_float_tensor(next_state)
            _, next_value = self.curr_agent(next_state)
            gen = self.memory.torch_gae_generator(next_value.item(), self.config.batch_size, \
                                            self.config.gamma, self.config.lam, \
                                            self.config.st_advantage, self.device)
            for states, actions, log_probs, advantages, returns in gen:
                ppo_update(self.curr_agent, self.optimizer, states, actions, log_probs, \
                           advantages, returns, config.num_epochs, config.eps_clip, \
                           config.entropy_coef, config.value_w)
            self.counter = 0
            self.memory.clear()
