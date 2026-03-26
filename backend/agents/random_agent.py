"""
Agent aléatoire — baseline de référence.
Choisit une action uniformément au hasard à chaque step.
"""
from env.frozen_lake import FrozenLakeEnv
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env: FrozenLakeEnv):
        super().__init__(env, name="Random")

    def select_action(self, state: int) -> int:
        return self.env.sample_action()
