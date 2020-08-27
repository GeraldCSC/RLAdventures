import random
import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, S,A,R, S_, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #i.e expand the list so position index works
        self.buffer[self.position] = (S,A,R,S_, done)
        self.position = (self.position + 1) % self.capacity #loop back if full

    def sample(self, batch_size): 
        batch = random.sample(self.buffer, batch_size)
        info_list = list(zip(*batch)) #this is essentially calling zip((SARS),(SARS)) so we get an list that returns all the S all A and so on..
        info_list[3] = [x for x in info_list[3] if x is not None] #third index is S_
        state, action, reward, next_state, done = map(np.stack, info_list) 
        return state, action[...,np.newaxis], reward[...,np.newaxis], next_state, done[...,np.newaxis]

    def torch_samples(self, batch_size, device="cpu"):
        S, A, R, S_, done = self.sample(batch_size)
        S = torch.tensor(S, device = device, dtype = torch.float32)
        S_ = torch.tensor(S_, device = device, dtype = torch.float32)
        A = torch.tensor(A, device = device, dtype = torch.long)
        R = torch.tensor(R, device = device, dtype = torch.float32)
        done = torch.tensor(done, device=device, dtype= torch.bool)
        return S,A,R,S_,done


    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    x = ReplayBuffer()
    u = np.random.randn(128)
    x.push(u, 2, 3,None, True)
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
