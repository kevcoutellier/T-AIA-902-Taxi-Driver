"""
sarsa.py — Agent SARSA (State-Action-Reward-State-Action).

SARSA est un algorithme ON-POLICY et TD(0).
Différence clé avec Q-Learning :

  Q-Learning (off-policy) :
    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
    → utilise la MEILLEURE action possible dans s' (greedy)

  SARSA (on-policy) :
    Q(s,a) ← Q(s,a) + α [ r + γ · Q(s',a') − Q(s,a) ]
    → utilise l'action a' RÉELLEMENT choisie par la politique ε-greedy

Conséquence : SARSA tient compte de l'exploration dans ses mises à jour.
Il converge vers une politique plus prudente/conservatrice que Q-Learning,
surtout dans les environnements avec des pénalités importantes.
"""

import numpy as np
from tqdm import tqdm
from environment import create_env, get_env_info


class SARSAAgent:
    """
    Agent SARSA avec stratégie ε-greedy.

    Identique à QLearningAgent en termes d'architecture (Q-table),
    mais la mise à jour utilise l'action suivante EFFECTIVE, pas le max.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table       = np.zeros((n_states, n_actions))
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """Stratégie ε-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def select_best_action(self, state):
        """Meilleure action connue (pour le test, sans exploration)."""
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, next_action):
        """
        Mise à jour SARSA : utilise Q(s', a') avec a' déjà choisi.

        Contrairement à Q-Learning qui calcule max Q(s', ·),
        ici on utilise la valeur Q de l'action qu'on VA réellement prendre.
        Cela rend la mise à jour cohérente avec la politique suivie.
        """
        current_q = self.q_table[state, action]
        next_q    = self.q_table[next_state, next_action]
        target    = reward + self.gamma * next_q
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def _find_convergence_episode(rewards, window=50, threshold=0.25):
    success = np.array(rewards) > 0
    for i in range(window, len(success) + 1):
        if np.mean(success[i - window : i]) >= threshold:
            return i - window + 1
    return None


def train_sarsa(n_episodes, alpha=0.1, gamma=0.99, epsilon=1.0,
                epsilon_min=0.01, epsilon_decay=0.995, verbose=True):
    """
    Entraîne un agent SARSA sur n_episodes.

    Boucle différente de Q-Learning : on doit choisir a' AVANT la mise à jour,
    puis réutiliser cette même a' comme action au pas suivant.

      s, a = init
      Pour chaque pas :
        1. Exécuter a → (r, s')
        2. Choisir a' depuis s' (ε-greedy)
        3. Q(s,a) ← Q(s,a) + α [r + γ·Q(s',a') − Q(s,a)]
        4. s, a = s', a'
    """
    env = create_env(render_mode=None)
    n_states, n_actions = get_env_info(env)

    agent = SARSAAgent(
        n_states, n_actions,
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
    )

    history = {"rewards": [], "steps": [], "epsilons": []}
    it = tqdm(range(n_episodes), desc="Training SARSA") if verbose else range(n_episodes)

    for _ in it:
        state, _ = env.reset()
        action = agent.select_action(state)   # a₀ choisi avant la boucle
        total_reward, steps = 0, 0

        for _ in range(200):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.select_action(next_state)   # a' choisi ici

            agent.learn(state, action, reward, next_state, next_action)

            total_reward += reward
            steps += 1
            state  = next_state
            action = next_action   # réutilisation de a' au prochain pas

            if terminated or truncated:
                break

        agent.decay_epsilon()
        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["epsilons"].append(agent.epsilon)

    env.close()
    history["convergence_episode"] = _find_convergence_episode(history["rewards"])
    return agent, history


def test_sarsa(agent, n_episodes, verbose=True):
    """Teste un agent SARSA entraîné (sans exploration)."""
    env = create_env(render_mode=None)
    all_rewards, all_steps, all_illegal = [], [], []
    it = tqdm(range(n_episodes), desc="Testing SARSA") if verbose else range(n_episodes)

    for _ in it:
        state, _ = env.reset()
        total_r, steps, illegal = 0, 0, 0
        for _ in range(200):
            action = agent.select_best_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps += 1
            if reward == -10:
                illegal += 1
            if terminated or truncated:
                break
        all_rewards.append(total_r)
        all_steps.append(steps)
        all_illegal.append(illegal)

    env.close()
    rewards   = np.array(all_rewards)
    steps_arr = np.array(all_steps)
    success   = rewards > 0
    return {
        "rewards": all_rewards,
        "steps":   all_steps,
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "mean_steps":      float(np.mean(steps_arr)),
        "std_steps":       float(np.std(steps_arr)),
        "success_rate":    float(np.mean(success) * 100),
        "reward_per_step": float(np.sum(rewards) / np.sum(steps_arr)),
        "illegal_actions": int(np.sum(all_illegal)),
        "convergence_episode": None,
    }
