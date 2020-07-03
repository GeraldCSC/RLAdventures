import gym
from attrdict import AttrDict
from helper import select_action, decompose_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Breakout-v0')

def main(args):
    memory = Replay(args.num_memory)
    loss_f = nn.SmoothL1Loss()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())
    for e in range(args.num_epoch):
        curr_obs = env.reset()
        prev_obs = None
        steps = e
        while True:
            for_input = curr_obs - prev_obs if prev_obs is not None else curr_obs
            action = select_action(policy_net, for_input, steps, args.eps_start, args.eps_decay, device)
            curr_obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    args = AttrDict()
    args_dict = {
        'batch_size' : 128,
        'num_epochs' : 300,
        'target_update' : 10,
        'gamma': 0.999,
        'num_memory': 300,
        'eps_start': 0.9,
        'eps_decay': 200,
        'gather_interval' : 200,
        'render': True
    }
    args.update(args_dict)
    main(args)



