import random
import time
import gymnasium as gym


def evaluate(agent, n_episodes=100, display_episodes=3):
    rewards = []
    steps = []
    successes = 0
    illegal_actions = 0

    display_indices = set()
    if display_episodes > 0 and n_episodes > 0:
        display_indices = set(random.sample(
            range(n_episodes), min(display_episodes, n_episodes)))

    for ep in range(n_episodes):
        render_mode = "human" if ep in display_indices else None
        env = gym.make("Taxi-v3", render_mode=render_mode)
        state, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.select_action(state, info)
            next_state, reward, done, truncated, info = env.step(action)
            if reward == -10:
                illegal_actions += 1
            state = next_state
            total_reward += reward
            step_count += 1

        if done and not truncated:
            successes += 1

        rewards.append(total_reward)
        steps.append(step_count)
        env.close()

    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    mean_steps = sum(steps) / len(steps) if steps else 0
    success_rate = successes / n_episodes if n_episodes > 0 else 0

    return {
        "rewards": rewards,
        "steps": steps,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "success_rate": success_rate,
        "illegal_actions": illegal_actions,
    }
