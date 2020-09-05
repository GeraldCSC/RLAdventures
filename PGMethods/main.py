from tqdm import tqdm

def run(trainer):
    env = trainer.env
    ep_num, reward_list = 0, []
    curr_state = env.reset()
    pbar = tqdm(range(trainer.max_frames))
    for frames in pbar:
        action = trainer.act(curr_state)
        next_state, reward, done, _ = env.step(action)
        trainer.update(reward, next_state,done)
        reward_list.append(reward)
        if done:
            ep_reward = sum(reward_list) 
            if frames % 1000 == 0:
                toshow = f"Episode Number: {ep_num} Current frame: {frames:.3e} Episode Rewards: {ep_reward}"
                pbar.set_description(toshow)
            curr_state = env.reset()
            ep_num += 1
            reward_list = []
        else:
            curr_state = next_state

if __name__ == "__main__":
    from VanillaTrainer import get_trainer
    trainer = get_trainer(lr=3e-4,num_epochs=20,render=False,monitor = False)
    run(trainer)
