from trainerbase import TrainerConfig, Trainer
from collections import namedtuple
from torch.optim import Adam
from utils.common import zero_grad, reward_to_go
from models.ALinear import LinearAgent
import torch

def get_trainer(**kwargs):
    config = VanillaConfig(**kwargs)
    return VanillaTrainer(LinearAgent, config)

Tracker = namedtuple('Tracker', ['state', 'act', 'log_prob', 'value'])

class VanillaConfig(TrainerConfig):
    max_frames = 1000000
    num_epochs = 3 #amount of updates on the same data
    num_steps = 1000 #amount of data we collect before update
    batch_size = 128
    lr = 1e-2
    render = False
    monitor = True
    vid_interval = 1000
    vid_save_path = "videos/"
    save_path = "models/model.pt"
    env_id = "CartPole-v1"
    gamma = 0.99
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class VanillaTrainer(Trainer): 
    def __init__(self, policyclass, config):
        super().__init__(config)
        self.agent = policyclass(input_size=self.obs_space, action_space=self.a_space).to(self.device)
        self.optimizer = Adam(self.curr_agent.parameters(), lr = config.lr)
        self.tracker = None
        self.counter = 0 

    def act(self, x):
        torch_obs = self.get_float_tensor(x)
        dist, value = self.agent.get_action(torch_obs)
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
