from agents.bruteforce import BruteForceAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent

AGENT_REGISTRY = {
    "bruteforce": BruteForceAgent,
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "dqn": DQNAgent,
}
