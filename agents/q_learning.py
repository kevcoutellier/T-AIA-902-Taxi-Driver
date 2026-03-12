import numpy as np
from agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.9995, epsilon_min=0.01,
                 n_states=500, n_actions=6):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))

    @property
    def name(self) -> str:
        return "Q-Learning"

    def select_action(self, state: int, info: dict = None) -> int:
        if np.random.random() < self.epsilon:
            if info and "action_mask" in info:
                valid = np.where(info["action_mask"] == 1)[0]
                return int(np.random.choice(valid))
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float,
              next_state: int, done: bool, info: dict = None) -> None:
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next * (1 - int(done))
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        if done:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

    def get_params(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }
