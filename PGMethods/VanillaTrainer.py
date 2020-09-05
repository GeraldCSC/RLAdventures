from trainerbase import TrainerConfig, Trainer
from torch.optim import Adam
from utils.common import zero_grad, reward_to_go
from utils.memory import PGMemory
from models.ALinear import LinearAgent
import torch

def get_trainer(**kwargs):
    config = VanillaConfig(**kwargs)
    return VanillaTrainer(LinearAgent, config)

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
        self.memory = PGMemory()
        self.counter = 0 

    def act(self, x):
        torch_obs = self.preproc_obs(x)
        action = self.agent.get_action(torch_obs)
        self.memory.push(state=x, action=action)
        return action

    def update(self, reward, next_state, done):
        self.memory.push(reward=reward, done=done)
        self.counter += 1
        if self.counter == self.config.num_steps:
            self.counter = 0
            self.memory.clear()
