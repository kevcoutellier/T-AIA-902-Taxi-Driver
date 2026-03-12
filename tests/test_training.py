import gymnasium as gym
from agents.bruteforce import BruteForceAgent
from training import train
from evaluation import evaluate


class TestTraining:
    def test_train_returns_metrics(self):
        agent = BruteForceAgent()
        metrics = train(agent, n_episodes=5, verbose=False)
        assert "rewards" in metrics
        assert "steps" in metrics
        assert len(metrics["rewards"]) == 5
        assert len(metrics["steps"]) == 5

    def test_train_time_limited(self):
        agent = BruteForceAgent()
        metrics = train(agent, time_limit=2.0, verbose=False)
        assert len(metrics["rewards"]) > 0
        assert metrics["training_time"] <= 3.0


class TestEvaluation:
    def test_evaluate_returns_metrics(self):
        agent = BruteForceAgent()
        metrics = evaluate(agent, n_episodes=5, display_episodes=0)
        assert "rewards" in metrics
        assert "steps" in metrics
        assert "success_rate" in metrics
        assert "mean_reward" in metrics
        assert "mean_steps" in metrics
        assert len(metrics["rewards"]) == 5
