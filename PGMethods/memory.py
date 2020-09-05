from common import gae
import torch

def torch_preproc(states, actions, log_probs, advantages, returns, device="cpu"):
    B = len(actions)
    states = torch.as_tensor(states, device = device, dtype = torch.float32)
    actions = torch.as_tensor(actions, device = device, dtype = torch.long).view(B, -1)
    f = lambda x: torch.as_tensor(x, device = device, dtype = torch.float32).view(B, -1)
    log_probs, advantages, returns = map(f, (log_probs, advantages, returns)) 
    return states, actions, log_probs, advantages, returns 

class PGMemory():
    def __init__(self):
        self.buf = []

    def clear(self):
        self.__init__()

    def push(self, s, a, log_prob, value, reward, done):
        self.buf.append((s,a,log_prob, value,reward,done))

    def torch_gae_generator(self,next_value, mini_batch_size, gamma = 0.99, lam=0.95, \
                            st_advantage = True,device="cpu"):
        states, actions, log_probs, values, rewards, dones = zip(*self.buf)
        advantages, returns = gae(next_value, list(values), rewards, dones, gamma, lam, st_advantage)
        states, actions, log_probs, advantages, returns = torch_preproc(states, actions, log_probs, advantages, returns)
        tot_size = len(self)
        num_sample = tot_size // mini_batch_size
        for _ in range(num_sample):
            idx = torch.randint(0, tot_size, (mini_batch_size,))
            yield states[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx]

    def __len__(self):
        return len(self.buf)
