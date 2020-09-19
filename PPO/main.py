"""
    runs the PPOTrainer on an openai gym like env interface
"""
def run(trainer):
    env = trainer.env
    ep_num, reward_list = 0, []
    curr_state = env.reset()
    pbar = range(int(trainer.max_frames))
    done_count = 0
    for frames in pbar:
        action = trainer.act(curr_state)
        next_state, reward, done, _ = env.step(action)
        trainer.update(reward, next_state,done)
        reward_list.append(reward)
        if done:
            done_count += 1
            ep_reward = sum(reward_list) 
            if done_count % 10 == 0:
                toshow = f"Episode Number: {ep_num} Current frame: {frames:.3e} Episode Rewards: {ep_reward}"
                print(toshow)
            if ep_reward > 220:
                trainer.save_model()
                return
            curr_state = env.reset()
            ep_num += 1
            reward_list = []
        else:
            curr_state = next_state
    trainer.save_model()

if __name__ == "__main__":
    from PPOTrainer import get_trainer
    trainer = get_trainer(value_w = 0.5,max_frames = 1e7,env_id = "LunarLander-v2",render=False, monitor=True, vid_interval = 100)
    run(trainer)
