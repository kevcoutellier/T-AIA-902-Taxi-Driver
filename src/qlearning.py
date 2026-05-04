"""
qlearning.py — Agent Q-Learning tabulaire.

Q-Learning est un algorithme OFF-POLICY :
- Il apprend la politique optimale indépendamment de la politique
  qu'il suit pour explorer (ε-greedy).

L'algorithme maintient une Q-table de taille (n_states × n_actions).
Chaque cellule Q(s, a) estime le reward total espéré si on fait
l'action 'a' dans l'état 's' puis qu'on agit optimalement ensuite.

Formule de mise à jour (Bellman) :
  Q(s, a) ← Q(s, a) + α × [r + γ × max(Q(s', a')) − Q(s, a)]
"""

import numpy as np
from tqdm import tqdm
from environment import create_env, get_env_info, run_episode


class QLearningAgent:
    """
    Agent Q-Learning avec stratégie ε-greedy.

    Attributs:
        q_table       : tableau numpy (500, 6) — le "cerveau" de l'agent
        alpha         : taux d'apprentissage
        gamma         : facteur de discount (importance du futur)
        epsilon       : probabilité d'exploration
        epsilon_min   : epsilon ne descend jamais en-dessous
        epsilon_decay : facteur de réduction d'epsilon après chaque épisode
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialise l'agent avec une Q-table remplie de zéros.

        Au début, l'agent ne sait rien : toutes les valeurs Q sont à 0.
        Cela signifie qu'aucune action n'est préférée — il va explorer
        uniformément grâce à ε=1.0.

        Args:
            n_states      : 500 (Taxi-v3)
            n_actions     : 6 (Taxi-v3)
            alpha         : learning rate — contrôle la vitesse d'apprentissage
            gamma         : discount factor — importance des rewards futurs
            epsilon       : taux d'exploration initial (1.0 = 100% random)
            epsilon_min   : taux d'exploration minimum
            epsilon_decay : multiplicateur appliqué à epsilon après chaque épisode
        """
        self.q_table = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """
        Stratégie ε-greedy :
        - Avec probabilité ε : action aléatoire (EXPLORATION)
        - Sinon             : action avec le meilleur Q (EXPLOITATION)

        Au début (ε=1.0), l'agent explore à 100%.
        Progressivement, ε diminue et l'agent exploite de plus en plus
        ses connaissances.

        Args:
            state: l'état actuel (un entier entre 0 et 499)

        Returns:
            L'action choisie (entier entre 0 et 5)
        """
        if np.random.random() < self.epsilon:
            # EXPLORATION : on essaie une action au hasard
            # Cela permet de découvrir de nouveaux chemins
            return np.random.randint(self.n_actions)
        else:
            # EXPLOITATION : on choisit l'action avec la plus haute valeur Q
            # C'est ce que l'agent pense être la meilleure action
            # np.argmax retourne l'indice du max dans le tableau
            return np.argmax(self.q_table[state])

    def select_best_action(self, state):
        """
        Choisit toujours la meilleure action connue (pas d'exploration).
        Utilisé uniquement pendant le TEST — on veut voir les vraies
        performances sans perturbation aléatoire.

        Args:
            state: l'état actuel

        Returns:
            L'action avec la plus haute valeur Q
        """
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """
        Met à jour la Q-table avec la formule de Bellman.

        C'est ICI que l'apprentissage se passe.

        La logique :
        1. On calcule la CIBLE : reward + γ × meilleur Q futur
           → C'est ce que la valeur Q DEVRAIT être
        2. On calcule l'ERREUR : cible − valeur actuelle
           → C'est à quel point on s'est trompé
        3. On ajuste Q d'un pas α dans la direction de la cible
           → On corrige progressivement notre estimation

        Args:
            state      : état dans lequel on était
            action     : action qu'on a faite
            reward     : récompense reçue
            next_state : état dans lequel on est arrivé
        """
        # Étape 1 : Quelle est la meilleure valeur Q dans le prochain état ?
        # C'est ce qu'on espère gagner à partir du prochain état
        best_next_q = np.max(self.q_table[next_state])

        # Étape 2 : La cible = reward immédiat + reward futur (pondéré par γ)
        target = reward + self.gamma * best_next_q

        # Étape 3 : L'erreur = écart entre cible et estimation actuelle
        current_q = self.q_table[state, action]
        error = target - current_q

        # Étape 4 : Mise à jour — on rapproche Q de la cible
        # Si α=0.1, on fait 10% du chemin vers la cible à chaque fois
        self.q_table[state, action] = current_q + self.alpha * error

    def decay_epsilon(self):
        """
        Réduit epsilon après chaque épisode d'entraînement.

        Formule : ε = max(ε × decay, ε_min)

        Avec decay=0.995 :
        - Épisode 0   : ε = 1.000 (100% exploration)
        - Épisode 100 : ε = 0.606 (60% exploration)
        - Épisode 500 : ε = 0.082 (8% exploration)
        - Épisode 1000: ε = 0.010 (1% exploration, minimum atteint)

        L'agent passe graduellement de l'exploration à l'exploitation.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def _find_convergence_episode(rewards, window=50, threshold=0.25):
    """Premier épisode où le taux de succès sur les 50 derniers épisodes atteint 25%."""
    success = np.array(rewards) > 0
    for i in range(window, len(success) + 1):
        if np.mean(success[i - window:i]) >= threshold:
            return i - window + 1  # 1-indexé, premier épisode de la fenêtre
    return None


def train_qlearning(n_episodes, alpha=0.1, gamma=0.99, epsilon=1.0,
                    epsilon_min=0.01, epsilon_decay=0.995, verbose=True):
    """
    Entraîne un agent Q-Learning sur n_episodes.

    La boucle d'entraînement :
    Pour chaque épisode :
      1. Reset l'environnement
      2. Tant que pas terminé :
         a. Choisir action (ε-greedy)
         b. Exécuter action → observer (s', r, done)
         c. Mettre à jour Q(s, a) avec la formule de Bellman
         d. s = s'
      3. Réduire epsilon

    Args:
        n_episodes   : nombre d'épisodes d'entraînement
        alpha        : learning rate
        gamma        : discount factor
        epsilon      : exploration initiale
        epsilon_min  : exploration minimale
        epsilon_decay: vitesse de réduction de l'exploration
        verbose      : afficher la barre de progression

    Returns:
        agent   : l'agent entraîné (avec sa Q-table remplie)
        history : dict avec les métriques d'entraînement par épisode
    """
    env = create_env(render_mode=None)
    n_states, n_actions = get_env_info(env)

    agent = QLearningAgent(
        n_states, n_actions,
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    # Historique pour les graphiques
    history = {
        "rewards": [],
        "steps": [],
        "epsilons": [],
    }

    iterator = tqdm(range(n_episodes), desc="Training Q-Learning") if verbose else range(n_episodes)

    for episode in iterator:
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        # Boucle interne : jouer un épisode complet
        for step in range(200):  # max 200 pas par épisode
            # 1. Choisir une action (ε-greedy)
            action = agent.select_action(state)

            # 2. Exécuter l'action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 3. APPRENDRE de cette expérience
            #    C'est la différence clé avec le brute-force :
            #    ici on met à jour la Q-table
            agent.learn(state, action, reward, next_state)

            total_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        # 4. Réduire epsilon (moins d'exploration au fil du temps)
        agent.decay_epsilon()

        # Sauvegarder les métriques de cet épisode
        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["epsilons"].append(agent.epsilon)

    env.close()
    history["convergence_episode"] = _find_convergence_episode(history["rewards"])
    return agent, history


def test_qlearning(agent, n_episodes, verbose=True):
    """
    Teste un agent entraîné sur n_episodes SANS exploration.

    Pendant le test, l'agent utilise select_best_action (pas de random).
    On mesure ses performances réelles.

    Args:
        agent      : agent Q-Learning déjà entraîné
        n_episodes : nombre d'épisodes de test
        verbose    : afficher la progression

    Returns:
        results: dict avec les métriques de test
    """
    env = create_env(render_mode=None)

    all_rewards, all_steps, all_illegal = [], [], []

    iterator = tqdm(range(n_episodes), desc="Testing Q-Learning") if verbose else range(n_episodes)

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
        "convergence_episode": None,  # rempli depuis history après train
    }


def display_qlearning_episode(agent):
    """
    Affiche un épisode de test dans le terminal.
    Permet de VOIR le taxi se déplacer intelligemment.
    """
    env = create_env(render_mode="ansi")

    state, _ = env.reset()
    total_reward = 0

    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    print("\n=== Episode Q-Learning (agent entraîné) ===\n")

    for step in range(50):
        action = agent.select_best_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        frame = env.render()
        print(f"Step {step + 1} | Action: {action_names[action]} | "
              f"Reward: {reward} | Total: {total_reward}")
        print(frame)

        if terminated or truncated:
            print(f"\nEpisode terminé en {step + 1} steps | "
                  f"Reward total: {total_reward}")
            break

    env.close()
