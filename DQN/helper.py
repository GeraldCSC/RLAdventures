import random
import torch

def select_action(model, state, t, EPS_START, EPS_DECAY, device):
    sample = random.random()
    eps_thresh = EPS_START * exp(-1 * t / EPS_DECAY)
    if sample > eps_thresh:
        state = torch.tensor(np.swapaxes(state, -1,0)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            return model(state).max(1)[1].item()
    else:
        return random.randint(0,3)

def decompose_sample(boot_strap_model,samples, device):
    states = []
    actions = []
    rewards = []
    boot_strap_value_tensor = torch.ones(len(samples),1,device = device)
    dones = []
    for i in range(len(samples)):
        item = samples[i]
        states.append([item.state])
        actions.append([item.action])
        rewards.append([item.reward])
        if item.next_state != None:
            next_states.append([item.next_state])
    states_tensor = torch.tensor(states, device = device, dtype = torch.float32)
    actions_tensor = torch.tensor(actions, device = device, dtype = torch.long)
    rewards_tensor = torch.tensor(rewards, device=device)
    next_state_tensor = torch.tensor(next_states, device = device, dtype = torch.float32)
    with torch.no_grad():
        output = boot_strap_model(next_state_tensor).max(1)[0]
    j = 0
    for i in range(len(boot_strap_value_tensor)):
        item = boot_strap_value_tensor[i]
        if item == 1:
            boot_strap_value_tensor[i] = output[j]
            j += 1
    return states_tensor, actions_tensor, rewards_tensor, boot_strap_value_tensor
