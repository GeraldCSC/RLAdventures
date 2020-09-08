import numpy as np
def get_sampler(total_size, batch_size, *args):
    for _ in range(total_size//batch_size):
        rand_idx = np.random.randint(0, total_size, batch_size)
        ret_list = []
        for item in args:
            ret_list.append(item[rand_idx])
        yield ret_list
