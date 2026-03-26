"""
Q-Learning — modèle-free, off-policy.

Mise à jour de la Q-table :
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

L'agent suit une politique ε-greedy pendant l'entraînement.
"""
import numpy as np
from env.frozen_lake import FrozenLakeEnv
from agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Paramètres
    ----------
    alpha   : taux d'apprentissage (learning rate)
    gamma   : facteur d'actualisation (discount factor)
    epsilon : probabilité d'exploration initiale
    epsilon_min   : valeur plancher d'epsilon
    epsilon_decay : multiplicateur appliqué à epsilon après chaque épisode
    """

    def __init__(
        self,
        env: FrozenLakeEnv,
        alpha: float = 0.8,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(env, name="Q-Learning")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table initialisée à zéro : shape (n_states, n_actions)
        self.Q = np.zeros((env.n_states, env.n_actions))

    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """ε-greedy : explore avec proba ε, exploite sinon."""
        if np.random.rand() < self.epsilon:
            return self.env.sample_action()
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: int) -> int:
        """Toujours greedy — utilisé en phase de test."""
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        """Mise à jour de la Q-table (Bellman off-policy)."""
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, n_episodes: int):
        """Override pour décroître epsilon après chaque épisode."""
        result = super().train(n_episodes)
        return result

    def run_episode(self, train: bool = False):
        ep = super().run_episode(train=train)
        if train:
            # Décroissance d'epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return ep

    def test(self, m_episodes: int):
        """Phase de test : on force greedy (epsilon = 0)."""
        saved_eps = self.epsilon
        self.epsilon = 0.0
        result = super().test(m_episodes)
        self.epsilon = saved_eps
        return result

    def reset_qtable(self):
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))
        self.epsilon = 1.0
