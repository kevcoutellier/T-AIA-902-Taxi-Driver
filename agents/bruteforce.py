import numpy as np
from agents.base import BaseAgent


class BruteForceAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "BruteForce"

    def select_action(self, state: int, info: dict = None) -> int:
        if info and "action_mask" in info:
            valid = np.where(info["action_mask"] == 1)[0]
            return int(np.random.choice(valid))
        return int(np.random.randint(0, 6))

    def learn(self, state, action, reward, next_state, done, info=None):
        pass

    def get_params(self) -> dict:
        return {}
