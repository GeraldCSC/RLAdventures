import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer():
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, S,A,R, S_, done):
        self.buffer.append((S,A,R,S_, done))

    def sample(self, batch_size): 
        batch = random.sample(self.buffer, batch_size)
        info_list = zip(*batch)
        state, action, reward, next_state, done = map(np.stack, info_list) 
        return state, action[...,np.newaxis], reward[...,np.newaxis], next_state, done[...,np.newaxis]

    def torch_samples(self, batch_size, device="cpu"):
        S, A, R, S_, done = self.sample(batch_size)
        S = torch.as_tensor(S, device = device, dtype = torch.float32)
        S_ = torch.as_tensor(S_, device = device, dtype = torch.float32)
        A = torch.as_tensor(A, device = device, dtype = torch.long)
        R = torch.as_tensor(R, device = device, dtype = torch.float32)
        done = torch.as_tensor(done, device=device, dtype= torch.long)
        return S,A,R,S_,done

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    x = ReplayBuffer()
    u = np.random.randn(128)
    x.push(u, 2, 3,u, True)
    x.push(u, 2, 3,u, False)
    x.push(u, 2, 3,u, False)
    x.push(u, 2, 3,u, True)
    S, A, R, S_, done = x.torch_samples(4)
    print(S.shape)
    print(S_.shape)
    print(A.shape)
    print(R.shape)
    print(done)
    temp = torch.zeros_like(R)
    print(temp.shape)
    print(temp[done].shape)
    print(R)
    temp[done] = torch.tensor([999, 200]).float().flatten()
    print(temp)
