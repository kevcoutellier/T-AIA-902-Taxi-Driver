from agents.bruteforce import BruteForceAgent

try:
    from agents.q_learning import QLearningAgent
except ImportError:
    QLearningAgent = None

try:
    from agents.sarsa import SARSAAgent
except ImportError:
    SARSAAgent = None

try:
    from agents.dqn import DQNAgent
except ImportError:
    DQNAgent = None

AGENT_REGISTRY = {
    "bruteforce": BruteForceAgent,
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "dqn": DQNAgent,
}
