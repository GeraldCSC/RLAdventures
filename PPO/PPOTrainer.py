from trainerbase import TrainerConfig, Trainer
from torch.optim import Adam
from utils.common import zero_grad, gae, convert_to_torch
from utils.batch_sampler import get_sampler
from utils.memory import PGMemory
from models.ACLinear import ACLinear
import torch

def get_trainer(**kwargs):
    config = PPOConfig(**kwargs)
    return PPOTrainer(ACLinear, config)

class PPOConfig(TrainerConfig):
    max_frames = 1000000
    num_epochs = 3
    lr = 2.4e-4
    num_steps = 1000
    batch_size = 200
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
        self.agent = policyclass(input_size=self.obs_space, action_space=self.a_space).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr = config.lr)
        self.memory = PGMemory()
        self.counter = 0 

    def act(self, x):
        """
            Acts according to the output distribution
        """
        x = self.preproc_obs(x)
        torch_obs = convert_to_torch([torch.float32], self.device, x)
        dist, value = self.agent.get_dist_value(torch_obs)
        action = dist.sample().detach()
        log_prob = dist.log_prob(action).detach()
        value = value.detach()
        self.memory.push(state=x, action=action, log_prob=log_prob, value=value)
        return action.item()

    def update(self, reward, next_state, done):
        self.memory.push(reward=reward, done=done)
        self.counter += 1
        if self.counter == self.config.num_steps:
            next_state = convert_to_torch([torch.float32], self.device, next_state)
            _, next_value = self.agent.get_dist_value(next_state)
            states, actions, rewards, values, log_probs, dones = self.memory.get("state","action",\
                                                              "reward","value", "log_prob", "done")
            advantages, returns = gae(next_value.item(), values, rewards, dones, self.config.gamma, \
                                      self.config.lam, self.config.st_advantage)
            self.train(states, actions, returns, advantages, log_probs)
            self.counter = 0
            self.memory.clear()

    def train(self, states, actions, returns, advantages, log_probs):
        rets = convert_to_torch([torch.float32, torch.int32, torch.float32, torch.float32, torch.float32], self.device, states, actions, returns, advantages, log_probs)
        gen = get_sampler(len(actions), self.config.batch_size, *rets)
        for batch_items in gen:
            for ep in range(self.config.num_epochs):
                self.train_epoch(*batch_items)

    def train_epoch(self, states, actions, returns, advantages, log_probs):
        eps_clip, value_w, entropy_coef = self.config.eps_clip, self.config.value_w, self.config.entropy_coef
        zero_grad(self.agent)
        dist, value = self.agent.get_dist_value(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)
        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1- eps_clip, 1 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (value.flatten() - returns).pow(2).mean()
        loss = actor_loss + value_w*critic_loss + entropy_coef * entropy
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.agent.state_dict(), self.config.save_path)
