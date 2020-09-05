import gym
import torch

class TrainerConfig:
    max_frames = 100000
    num_steps = 1000
    batch_size = 256
    lr = 0.0001
    render = False
    monitor = True
    vid_interval = 1000
    vid_save_path = "videos/"
    save_path = "models/model.pt"
    env_id = "CartPole-v1"
    gamma = 0.99
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_id)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.a_space, self.obs_space = self.env.action_space.n, self.env.observation_space.shape[0]
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.max_frames = max_frames
        if config.render:
            self.env.render()
        if config.monitor:
            self.env = gym.wrappers.Monitor(self.env, config.vid_save_path, \
                                            video_callable = lambda ep: ep % config.vid_interval == 0,force= True)
    def update(self, reward, next_state, done):
        raise NotImplementedError

    def act(self, x):
        raise NotImplementedError
