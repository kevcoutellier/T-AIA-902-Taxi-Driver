from agents import AGENT_REGISTRY
from training import train
from evaluation import evaluate


def run_benchmark(train_episodes=5000, test_episodes=100, verbose=True):
    results = {}

    for agent_name, AgentClass in AGENT_REGISTRY.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Benchmarking: {agent_name}")
            print(f"{'='*50}")

        agent = AgentClass()
        train_metrics = train(agent, n_episodes=train_episodes, verbose=verbose)
        test_metrics = evaluate(agent, n_episodes=test_episodes, display_episodes=0)

        results[agent_name] = {
            "train": train_metrics,
            "test": test_metrics,
            "params": agent.get_params(),
            "agent_name": agent.name,
        }

        if verbose:
            print(f"  Mean Reward: {test_metrics['mean_reward']:.2f}")
            print(f"  Mean Steps:  {test_metrics['mean_steps']:.1f}")
            print(f"  Success Rate: {test_metrics['success_rate']*100:.1f}%")

    return results
