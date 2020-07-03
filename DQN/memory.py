from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Replay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, S,A,R,S_):
        if self.position < self.capacity:
            S = np.moveaxis(S, -1,0)
            S_ = np.moveaxis(S_, -1,0)
            self.memory.append(Transition(S,A,R,S_))
            self.position += 1

    def sample(self, batch_size): #maybe convert to tensors here?
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return self.position
