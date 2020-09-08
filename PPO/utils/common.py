import numpy as np
import torch
def zero_grad(model):
    for param in model.parameters():
        param.grad = None

def convert_to_torch(dtypes, device, *args):
    ret_list = []
    assert len(dtypes) == len(args)
    for item, d_type in zip(args, dtypes):
        converted = torch.as_tensor(item, device= device, dtype = d_type)
        if len(args) == 1:
            return converted
        ret_list.append(converted)
    return ret_list

def reward_to_go(rewards, dones, gamma = 1):
    """
        assuming that the rewards are put in chornological order
        rewards: list
        dones: list
        returns: an np array
    """
    curr_reward = 0
    discounted_rewards = np.zeros_like(rewards, dtype = "float32")
    for i in reversed(range(len(rewards))):
        mask = 1 - dones[i]
        curr_reward = gamma * curr_reward*mask + rewards[i]
        discounted_rewards[i] = curr_reward
    return discounted_rewards

def gae(next_value, values, rewards, dones, gamma = 0.99, lam = 0.95, standardize = False):
    """
        next_value: scalar for the last value, could be 0 or could be something if we sampled
            parts of a value
        value: list 
        rewards: list
        dones: list
    """
    assert len(values) == len(rewards) 
    values = values + [next_value]
    gae_array = np.zeros_like(rewards, dtype= "float32")
    returns = gae_array.copy()
    curr_gae = 0
    for i in reversed(range(len(rewards))):
        R, V_next, V_curr, mask = rewards[i], values[i+1], values[i], 1 - dones[i]
        delta = R + gamma*V_next*mask - V_curr 
        curr_gae = delta + gamma*lam*curr_gae*mask
        gae_array[i] = curr_gae
        returns[i] = curr_gae + V_curr
    if standardize:
        gae_array = (gae_array - gae_array.mean()) / (gae_array.std() + 1e-8)
    return gae_array, returns
