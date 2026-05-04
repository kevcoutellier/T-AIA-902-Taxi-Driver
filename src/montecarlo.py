"""
montecarlo.py — Agent Monte Carlo First-Visit.

Monte Carlo est un algorithme ON-POLICY et MODEL-FREE.
Contrairement a Q-Learning qui met a jour la Q-table a chaque pas,
Monte Carlo attend la FIN de l'episode pour mettre a jour.

Principe :
1. Jouer un episode complet en suivant la politique epsilon-greedy
2. Calculer le VRAI retour cumule G pour chaque paire (etat, action)
3. Mettre a jour Q(s, a) avec la moyenne des retours observes

Avantage : utilise le vrai retour, pas une estimation (pas de biais)
Inconvenient : doit attendre la fin de l'episode (plus lent a converger)

Variante : "First-Visit" = on ne met a jour que la PREMIERE fois
qu'un etat est visite dans un episode (evite le double-comptage).
"""

import numpy as np
from tqdm import tqdm
from environment import create_env, get_env_info, run_episode


class MonteCarloAgent:
    """
    Agent Monte Carlo First-Visit avec strategie epsilon-greedy.

    Differences avec Q-Learning :
    - Pas de learning rate alpha (on fait la moyenne exacte)
    - Pas de discount factor gamma dans la mise a jour
      (il est applique lors du calcul du retour G)
    - La mise a jour se fait apres l'episode, pas a chaque pas
    """

    def __init__(self, n_states, n_actions, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Args:
            n_states     : 500 (Taxi-v3)
            n_actions    : 6 (Taxi-v3)
            gamma        : discount factor pour le calcul du retour G
            epsilon      : taux d'exploration initial
            epsilon_min  : taux d'exploration minimum
            epsilon_decay: facteur de reduction d'epsilon
        """
        # Q-table : meme structure que Q-Learning
        self.q_table = np.zeros((n_states, n_actions))

        # Compteur de visites : combien de fois chaque (s, a) a ete vu
        # Utilise pour calculer la moyenne incrementale
        self.visit_counts = np.zeros((n_states, n_actions))

        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """
        Strategie epsilon-greedy (identique a Q-Learning).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def select_best_action(self, state):
        """
        Meilleure action connue (pour le test).
        """
        return np.argmax(self.q_table[state])

    def learn_from_episode(self, episode_history):
        """
        Met a jour la Q-table a partir d'un episode COMPLET.

        C'est la difference fondamentale avec Q-Learning :
        on attend d'avoir TOUT l'episode avant d'apprendre.

        Algorithme First-Visit Monte Carlo :
        1. On parcourt l'episode de la FIN vers le DEBUT
        2. On calcule le retour cumule G a chaque pas
        3. Pour chaque (s, a) visite pour la PREMIERE fois,
           on met a jour Q(s, a) avec la moyenne incrementale

        Args:
            episode_history: liste de tuples (state, action, reward)
        """
        # Retour cumule G, calcule en remontant depuis la fin
        G = 0

        # Ensemble des paires (s, a) deja visitees dans cet episode
        # Pour le First-Visit : on ne compte que la premiere occurrence
        visited = set()

        # On parcourt l'episode A L'ENVERS (du dernier pas au premier)
        for state, action, reward in reversed(episode_history):
            # Calcul du retour : G = r + gamma * G
            # En remontant : chaque reward est pondere par gamma^(distance a la fin)
            G = reward + self.gamma * G

            pair = (state, action)

            # First-Visit : on ne met a jour que si c'est la premiere visite
            # (en parcourant a l'envers, la derniere rencontree est la premiere)
            if pair not in visited:
                visited.add(pair)

                # Moyenne incrementale :
                # new_mean = old_mean + (new_value - old_mean) / n
                # C'est mathematiquement equivalent a recalculer la moyenne
                # de tous les retours, mais plus efficace en memoire
                self.visit_counts[state, action] += 1
                n = self.visit_counts[state, action]
                self.q_table[state, action] += (G - self.q_table[state, action]) / n

    def decay_epsilon(self):
        """Reduit epsilon apres chaque episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def _find_convergence_episode(rewards, window=50, threshold=0.25):
    success = np.array(rewards) > 0
    for i in range(window, len(success) + 1):
        if np.mean(success[i - window:i]) >= threshold:
            return i - window + 1
    return None


def train_montecarlo(n_episodes, gamma=0.99, epsilon=1.0,
                     epsilon_min=0.01, epsilon_decay=0.995, verbose=True):
    """
    Entraine un agent Monte Carlo sur n_episodes.

    La boucle est differente de Q-Learning :
    1. Jouer un episode COMPLET en enregistrant chaque (s, a, r)
    2. APRES l'episode, mettre a jour la Q-table
    3. Reduire epsilon

    Args:
        n_episodes   : nombre d'episodes d'entrainement
        gamma        : discount factor
        epsilon      : exploration initiale
        epsilon_min  : exploration minimale
        epsilon_decay: vitesse de reduction
        verbose      : afficher la progression

    Returns:
        agent   : l'agent entraine
        history : dict avec les metriques par episode
    """
    env = create_env(render_mode=None)
    n_states, n_actions = get_env_info(env)

    agent = MonteCarloAgent(
        n_states, n_actions,
        gamma=gamma, epsilon=epsilon,
        epsilon_min=epsilon_min, epsilon_decay=epsilon_decay
    )

    history = {
        "rewards": [],
        "steps": [],
        "epsilons": [],
    }

    iterator = tqdm(range(n_episodes), desc="Training Monte Carlo") if verbose else range(n_episodes)

    for episode in iterator:
        state, _ = env.reset()
        episode_history = []  # On stocke TOUT l'episode
        total_reward = 0
        steps = 0

        # Phase 1 : JOUER l'episode complet
        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # On enregistre chaque transition (pas de mise a jour ici !)
            episode_history.append((state, action, reward))

            total_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Phase 2 : APPRENDRE de l'episode complet
        # C'est seulement ICI que la Q-table est mise a jour
        agent.learn_from_episode(episode_history)

        # Phase 3 : Reduire epsilon
        agent.decay_epsilon()

        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["epsilons"].append(agent.epsilon)

    env.close()
    history["convergence_episode"] = _find_convergence_episode(history["rewards"])
    return agent, history


def test_montecarlo(agent, n_episodes, verbose=True):
    """
    Teste un agent Monte Carlo entraine (sans exploration).
    """
    env = create_env(render_mode=None)

    all_rewards, all_steps, all_illegal = [], [], []

    iterator = tqdm(range(n_episodes), desc="Testing Monte Carlo") if verbose else range(n_episodes)

    for _ in iterator:
        state, _ = env.reset()
        total_reward, steps, illegal = 0, 0, 0
        for _ in range(200):
            action = agent.select_best_action(state)
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


def display_montecarlo_episode(agent):
    """Affiche un episode de test dans le terminal."""
    env = create_env(render_mode="ansi")
    state, _ = env.reset()
    total_reward = 0

    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    print("\n=== Episode Monte Carlo (agent entraine) ===\n")

    for step in range(50):
        action = agent.select_best_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        frame = env.render()
        print(f"Step {step + 1} | Action: {action_names[action]} | "
              f"Reward: {reward} | Total: {total_reward}")
        print(frame)

        if terminated or truncated:
            print(f"\nEpisode termine en {step + 1} steps | "
                  f"Reward total: {total_reward}")
            break

    env.close()
