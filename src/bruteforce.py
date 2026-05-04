"""
bruteforce.py — Agent brute-force (actions aléatoires).

Cet agent sert de BASELINE : c'est le pire algo possible.
Tout algorithme RL doit faire mieux que lui pour être considéré utile.

Résultat attendu : ~200 steps par épisode, reward très négatif.
"""

import numpy as np
from tqdm import tqdm
from environment import create_env, run_episode


class BruteForceAgent:
    """
    Agent qui choisit une action au hasard à chaque pas.

    Il n'apprend rien. Il n'a pas de mémoire.
    C'est l'équivalent d'un humain qui appuierait sur des boutons au hasard.
    """

    def __init__(self, n_actions=6):
        """
        Args:
            n_actions: nombre d'actions possibles (6 pour Taxi-v3)
        """
        self.n_actions = n_actions

    def select_action(self, state):
        """
        Choisit une action complètement aléatoire.
        Le paramètre state est ignoré — l'agent ne regarde même pas
        où il est.

        Args:
            state: l'état actuel (ignoré)

        Returns:
            Une action aléatoire entre 0 et 5
        """
        return np.random.randint(self.n_actions)


def run_bruteforce(n_episodes):
    """
    Lance n_episodes parties avec l'agent brute-force et collecte les 8 métriques.
    """
    env = create_env(render_mode=None)
    agent = BruteForceAgent()

    all_rewards, all_steps, all_illegal = [], [], []

    for _ in tqdm(range(n_episodes), desc="Brute-force"):
        state, _ = env.reset()
        total_reward, steps, illegal = 0, 0, 0
        for _ in range(500):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if reward == -10:
                illegal += 1
            if terminated or truncated:
                break
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_illegal.append(illegal)

    env.close()

    rewards = np.array(all_rewards)
    steps_arr = np.array(all_steps)
    success = rewards > 0

    return {
        "rewards": all_rewards,
        "steps": all_steps,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "mean_steps":       float(np.mean(steps_arr)),
        "std_steps":        float(np.std(steps_arr)),
        "success_rate":     float(np.mean(success) * 100),
        "reward_per_step":  float(np.sum(rewards) / np.sum(steps_arr)),
        "illegal_actions":  int(np.sum(all_illegal)),
        "convergence_episode": None,
    }


def display_bruteforce_episode():
    """
    Joue et affiche un épisode brute-force dans le terminal.
    Permet de VOIR ce que fait un agent aléatoire (spoiler : c'est chaotique).
    """
    env = create_env(render_mode="ansi")
    agent = BruteForceAgent()

    state, _ = env.reset()
    total_reward = 0

    print("\n=== Episode Brute-Force ===\n")

    for step in range(100):  # On limite à 100 pas pour l'affichage
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Affiche la grille à chaque pas
        frame = env.render()
        print(f"Step {step + 1} | Action: {action} | Reward: {reward} | Total: {total_reward}")
        print(frame)

        if terminated or truncated:
            print(f"\nEpisode terminé en {step + 1} steps | Reward total: {total_reward}")
            break

    env.close()
