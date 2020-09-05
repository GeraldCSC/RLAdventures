import numpy as np

def zero_grad(model):
    for param in model.parameters():
        param.grad = None

def reward_to_go(rewards, dones, gamma = 1):
    """
        assuming that the rewards are put in chornological order
    """
    curr_reward = 0
    discounted_rewards = np.zeros_like(rewards, dtype = "float32")
    for i in reversed(range(len(rewards))):
        mask = 1 - dones[i]
        curr_reward = gamma * curr_reward*mask + rewards[i]
        discounted_rewards[i] = curr_reward
        if mask == 0:
            curr_reward = 0
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
    values.append(next_value)
    gae_array = np.zeros_like(rewards, dtype= "float32")
    returns = gae_array.copy()
    curr_gae = 0
    for i in reversed(range(len(rewards))):
        R, V_next, V_curr, mask = rewards[i], values[i+1], values[i], 1 - dones[i]
        delta = R + gamma*V_next*mask - V_curr 
        curr_gae = delta + gamma*lam*curr_gae*mask
        gae_array[i] = curr_gae
        ####
        if mask == 0: # i think this is necessary since I am resetting the environment on a done
            curr_gae = 0
        ####
        returns[i] = curr_gae + V_curr
    if standardize:
        gae_array = (gae_array - gae_array.mean()) / (gae_array.std() + 1e-8)
    return gae_array, returns
