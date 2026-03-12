import time
import gymnasium as gym


def train(agent, n_episodes=None, time_limit=None, verbose=True):
    env = gym.make("Taxi-v3")
    rewards = []
    steps = []
    start_time = time.time()
    episode = 0

    while True:
        if n_episodes is not None and episode >= n_episodes:
            break
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        state, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.select_action(state, info)
            next_state, reward, done, truncated, next_info = env.step(action)
            agent.learn(state, action, reward, next_state, done or truncated, info)
            state = next_state
            info = next_info
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)
        episode += 1

        if verbose and episode % 100 == 0:
            avg_r = sum(rewards[-100:]) / min(100, len(rewards[-100:]))
            avg_s = sum(steps[-100:]) / min(100, len(steps[-100:]))
            print(f"Episode {episode} | Avg Reward: {avg_r:.2f} | Avg Steps: {avg_s:.1f}")

    env.close()
    training_time = time.time() - start_time

    return {
        "rewards": rewards,
        "steps": steps,
        "training_time": training_time,
        "episodes": episode,
    }
