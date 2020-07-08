import numpy as np
import torch 

def get_boot_strap_value(target_net,nextstate, done):
    """
        Precondition:
            all inputs are torch tensors 
            rewards: N x 1
            nextstate: N x 3 x H x W
            done: N x 1
    """
    boot_straped_value = torch.zeros_like(done).float()
    done = done.flatten()
    nextstate = nextstate[done == False]
    with torch.no_grad():
        output = target_net(nextstate).max(1)[0]
    boot_straped_value[done == False] = output.unsqueeze(1)
    return boot_straped_value

def preproc(image):
    image = image.astype("float32")
    image = (image / 127.5) - 1
    image = image.transpose(-1,0,1)
    return image

