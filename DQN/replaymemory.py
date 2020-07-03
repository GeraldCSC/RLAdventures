from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class Replay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, S,A,R,S_):
        if self.position < self.capacity:
            self.memory.append(Transition(S,A,R,S_))
            self.position += 1
        else:
            print("Replay Memory overflow")

    def sample(self, batch_size): 
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return self.position
