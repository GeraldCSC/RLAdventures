import random
import numpy as np

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, S,A,R,S_, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #i.e expand the list so position index works
        self.buffer[self.position] = (S,A,R,S_, done)
        self.position = (self.position + 1) % self.capacity #loop back if full

    def sample(self, batch_size): 
        batch = random.sample(self.buffer, batch_size)
        batch_iterator = zip(*batch) #this is essentially calling zip((SARS),(SARS)) so we get an iterator that returns all the S all A and so on..
        state, action, reward, next_state, done = map(np.stack, batch_iterator) #this stacks all the S and A and so on together into matrices
        return state, action[...,np.newaxis], reward[...,np.newaxis], next_state, done[...,np.newaxis]

    def __len__(self):
        return len(self.buffer)
