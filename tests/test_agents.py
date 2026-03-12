import pytest
import numpy as np
from agents.base import BaseAgent


def test_base_agent_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseAgent()


def test_base_agent_requires_select_action():
    class Incomplete(BaseAgent):
        @property
        def name(self):
            return "incomplete"

        def learn(self, state, action, reward, next_state, done, info=None):
            pass

        def get_params(self):
            return {}

    with pytest.raises(TypeError):
        Incomplete()


from agents.bruteforce import BruteForceAgent


class TestBruteForceAgent:
    def test_name(self):
        agent = BruteForceAgent()
        assert agent.name == "BruteForce"

    def test_select_action_returns_valid_action(self):
        agent = BruteForceAgent()
        action = agent.select_action(0)
        assert 0 <= action <= 5

    def test_select_action_uses_action_mask(self):
        agent = BruteForceAgent()
        mask = np.array([1, 0, 0, 0, 0, 0])
        for _ in range(20):
            action = agent.select_action(0, info={"action_mask": mask})
            assert action == 0

    def test_learn_is_noop(self):
        agent = BruteForceAgent()
        agent.learn(0, 1, -1.0, 1, False)  # should not raise

    def test_get_params(self):
        agent = BruteForceAgent()
        assert agent.get_params() == {}
