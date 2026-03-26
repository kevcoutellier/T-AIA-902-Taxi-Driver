"""
Deep Q-Network (DQN) — approximation de la Q-table par un réseau de neurones.

Caractéristiques implémentées :
  - Réseau fully-connected (MLP) avec PyTorch
  - Experience Replay (replay buffer)
  - Target network (mis à jour toutes les C étapes)
  - Politique ε-greedy avec décroissance
  - Encodage one-hot de l'état (discret → vecteur)
"""
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.frozen_lake import FrozenLakeEnv
from agents.base import BaseAgent, EpisodeResult, TrainResult


# ------------------------------------------------------------------
# Réseau de neurones
# ------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP : input(one-hot state) → hidden → hidden → output(Q-values)."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Replay Buffer
# ------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------------
# Agent DQN
# ------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """
    Paramètres
    ----------
    gamma           : discount factor
    epsilon         : exploration initiale
    epsilon_min     : plancher epsilon
    epsilon_decay   : décroissance multiplicative
    lr              : learning rate Adam
    batch_size      : taille du mini-batch
    buffer_capacity : taille du replay buffer
    target_update   : fréquence (en épisodes) de synchro du target network
    hidden          : taille des couches cachées
    """

    def __init__(
        self,
        env: FrozenLakeEnv,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update: int = 10,
        hidden: int = 64,
    ):
        super().__init__(env, name="DQN")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(env.n_states, env.n_actions, hidden).to(self.device)
        self.target_net = QNetwork(env.n_states, env.n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Encodage one-hot
    # ------------------------------------------------------------------

    def _encode(self, state: int) -> torch.Tensor:
        v = torch.zeros(self.env.n_states, device=self.device)
        v[state] = 1.0
        return v

    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.sample_action()
        with torch.no_grad():
            q_vals = self.policy_net(self._encode(state).unsqueeze(0))
        return int(q_vals.argmax().item())

    def update(self, state, action, reward, next_state, done):
        """Pousse la transition dans le buffer, puis entraîne si assez de données."""
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.stack([self._encode(s) for s in states])
        next_states_t = torch.stack([self._encode(s) for s in next_states])
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a) courant
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target : r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_episode(self, train: bool = False) -> EpisodeResult:
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        t0 = time.time()

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if train:
                self.update(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        if train:
            self._episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Synchronisation du target network
            if self._episode_count % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

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
