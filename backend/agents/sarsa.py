"""
SARSA — State Action Reward State Action.
Algorithme on-policy : la mise à jour utilise l'action réellement choisie en s'.

Mise à jour :
    Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
    où a' est l'action sélectionnée par la politique depuis s'.
"""
import numpy as np
from env.frozen_lake import FrozenLakeEnv
from agents.base import BaseAgent, EpisodeResult, TrainResult
import time


class SARSAAgent(BaseAgent):
    """
    Paramètres
    ----------
    alpha   : learning rate
    gamma   : discount factor
    epsilon : exploration initiale (ε-greedy)
    epsilon_min   : plancher epsilon
    epsilon_decay : décroissance multiplicative par épisode
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
        super().__init__(env, name="SARSA")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((env.n_states, env.n_actions))

    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.sample_action()
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        """
        SARSA : on utilise l'action a' choisie via la politique ε-greedy,
        pas le max. Cette méthode n'est pas appelée directement ici
        (on override run_episode pour passer a').
        """

    # ------------------------------------------------------------------
    # Override complet de run_episode pour SARSA (on-policy)
    # ------------------------------------------------------------------

    def run_episode(self, train: bool = False) -> EpisodeResult:
        state = self.env.reset()
        action = self.select_action(state)
        total_reward = 0.0
        steps = 0
        t0 = time.time()

        while True:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if train:
                next_action = self.select_action(next_state)
                # Mise à jour SARSA (on-policy)
                q_next = 0.0 if done else self.Q[next_state, next_action]
                td_error = reward + self.gamma * q_next - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
            else:
                next_action = self.select_action(next_state)

            total_reward += reward
            steps += 1
            state = next_state
            action = next_action

            if done:
                break

        if train:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return EpisodeResult(
            reward=total_reward,
            steps=steps,
            won=total_reward > 0,
            duration=time.time() - t0,
        )

    def test(self, m_episodes: int) -> TrainResult:
        saved_eps = self.epsilon
        self.epsilon = 0.0
        result = super().test(m_episodes)
        self.epsilon = saved_eps
        return result

    def reset_qtable(self):
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))
        self.epsilon = 1.0
