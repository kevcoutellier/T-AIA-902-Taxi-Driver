import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent


class QNetwork(nn.Module):
    def __init__(self, n_states=500, n_actions=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, n_states=500, n_actions=6, lr=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.9995,
                 epsilon_min=0.01, batch_size=64,
                 buffer_capacity=10000, target_update=100):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(n_states, n_actions).to(self.device)
        self.target_net = QNetwork(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=buffer_capacity)
        self.steps = 0

    @property
    def name(self) -> str:
        return "DQN"

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return torch.tensor(one_hot, device=self.device).unsqueeze(0)

    def select_action(self, state: int, info: dict = None) -> int:
        if np.random.random() < self.epsilon:
            if info and "action_mask" in info:
                valid = np.where(info["action_mask"] == 1)[0]
                return int(np.random.choice(valid))
            return int(np.random.randint(0, self.n_actions))

        with torch.no_grad():
            q_values = self.policy_net(self._state_to_tensor(state))
            return int(q_values.argmax(dim=1).item())

    def learn(self, state: int, action: int, reward: float,
              next_state: int, done: bool, info: dict = None) -> None:
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.steps += 1

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_batch = torch.zeros(self.batch_size, self.n_states,
                                  device=self.device)
        next_state_batch = torch.zeros(self.batch_size, self.n_states,
                                       device=self.device)
        for i in range(self.batch_size):
            state_batch[i, states[i]] = 1.0
            next_state_batch[i, next_states[i]] = 1.0

        action_batch = torch.tensor(actions, device=self.device,
                                    dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=self.device,
                                    dtype=torch.float32)
        done_batch = torch.tensor(dones, device=self.device,
                                  dtype=torch.float32)

        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(dim=1)[0]
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if done:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

    def get_params(self) -> dict:
        return {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "batch_size": self.batch_size,
            "buffer_capacity": self.replay_buffer.maxlen,
            "target_update": self.target_update,
        }
