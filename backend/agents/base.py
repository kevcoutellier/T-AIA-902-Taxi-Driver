"""
Classe de base pour tous les agents RL.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from env.frozen_lake import FrozenLakeEnv


@dataclass
class EpisodeResult:
    reward: float
    steps: int
    won: bool
    duration: float  # secondes


@dataclass
class TrainResult:
    episodes: int
    episode_rewards: list = field(default_factory=list)
    episode_steps: list = field(default_factory=list)
    episode_wins: list = field(default_factory=list)
    train_duration: float = 0.0

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def avg_steps(self) -> float:
        return float(np.mean(self.episode_steps)) if self.episode_steps else 0.0

    @property
    def win_rate(self) -> float:
        return float(np.mean(self.episode_wins)) if self.episode_wins else 0.0

    def summary(self) -> str:
        return (
            f"  Episodes      : {self.episodes}\n"
            f"  Avg reward    : {self.avg_reward:.4f}\n"
            f"  Avg steps     : {self.avg_steps:.1f}\n"
            f"  Win rate      : {self.win_rate * 100:.1f}%\n"
            f"  Train time    : {self.train_duration:.2f}s"
        )


class BaseAgent(ABC):
    """Interface commune à tous les agents."""

    def __init__(self, env: FrozenLakeEnv, name: str = "Agent"):
        self.env = env
        self.name = name

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Retourne l'action à effectuer depuis l'état donné."""

    def run_episode(self, train: bool = False) -> EpisodeResult:
        """Joue un épisode complet et retourne le résultat."""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        t0 = time.time()

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            if train:
                self.update(state, action, reward, next_state, terminated or truncated)

            total_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        return EpisodeResult(
            reward=total_reward,
            steps=steps,
            won=total_reward > 0,
            duration=time.time() - t0,
        )

    def update(self, state, action, reward, next_state, done):
        """Override dans les agents entraînables."""

    def train(self, n_episodes: int) -> TrainResult:
        """Entraîne l'agent sur n épisodes."""
        result = TrainResult(episodes=n_episodes)
        t0 = time.time()

        for _ in range(n_episodes):
            ep = self.run_episode(train=True)
            result.episode_rewards.append(ep.reward)
            result.episode_steps.append(ep.steps)
            result.episode_wins.append(ep.won)

        result.train_duration = time.time() - t0
        return result

    def test(self, m_episodes: int) -> TrainResult:
        """Teste l'agent sur m épisodes (sans mise à jour)."""
        result = TrainResult(episodes=m_episodes)
        t0 = time.time()

        for _ in range(m_episodes):
            ep = self.run_episode(train=False)
            result.episode_rewards.append(ep.reward)
            result.episode_steps.append(ep.steps)
            result.episode_wins.append(ep.won)

        result.train_duration = time.time() - t0
        return result
