import argparse
import sys

from agents import AGENT_REGISTRY
from training import train
from evaluation import evaluate
from benchmark import run_benchmark
from report import generate_report


OPTIMIZED_PARAMS = {
    "q_learning": {"alpha": 0.1, "gamma": 0.99, "epsilon": 1.0,
                   "epsilon_decay": 0.999, "epsilon_min": 0.01},
    "sarsa": {"alpha": 0.1, "gamma": 0.99, "epsilon": 1.0,
              "epsilon_decay": 0.999, "epsilon_min": 0.01},
    "dqn": {"lr": 0.001, "gamma": 0.99, "epsilon": 1.0,
            "epsilon_decay": 0.999, "epsilon_min": 0.01,
            "batch_size": 64, "buffer_capacity": 10000, "target_update": 100},
    "bruteforce": {},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Taxi-v3 Reinforcement Learning Solver"
    )
    parser.add_argument("--mode", choices=["user", "time-limited", "benchmark"],
                        required=True, help="Running mode")
    parser.add_argument("--agent", choices=list(AGENT_REGISTRY.keys()),
                        help="Agent to use (not needed for benchmark)")
    parser.add_argument("--train", type=int, help="Number of training episodes")
    parser.add_argument("--test", type=int, required=True,
                        help="Number of testing episodes")
    parser.add_argument("--time", type=float,
                        help="Time limit in seconds (time-limited mode)")
    parser.add_argument("--display", type=int, default=3,
                        help="Number of random episodes to display during testing")

    # Hyperparameters for user mode
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)

    return parser.parse_args()


def run_user_mode(args):
    if not args.agent:
        print("Error: --agent is required for user mode")
        sys.exit(1)
    if not args.train:
        print("Error: --train is required for user mode")
        sys.exit(1)

    AgentClass = AGENT_REGISTRY[args.agent]
    if args.agent in ("q_learning", "sarsa"):
        agent = AgentClass(
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
        )
    elif args.agent == "dqn":
        agent = AgentClass(
            lr=args.lr, gamma=args.gamma, epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
            batch_size=args.batch_size,
        )
    else:
        agent = AgentClass()

    print(f"\nTraining {agent.name} for {args.train} episodes...")
    print(f"Parameters: {agent.get_params()}\n")
    train_metrics = train(agent, n_episodes=args.train)

    print(f"\nTesting {agent.name} for {args.test} episodes...")
    test_metrics = evaluate(agent, n_episodes=args.test,
                            display_episodes=args.display)

    print(f"\n{'='*40}")
    print(f"Results for {agent.name}")
    print(f"{'='*40}")
    print(f"Mean Reward:    {test_metrics['mean_reward']:.2f}")
    print(f"Mean Steps:     {test_metrics['mean_steps']:.1f}")
    print(f"Success Rate:   {test_metrics['success_rate']*100:.1f}%")
    print(f"Illegal Actions: {test_metrics['illegal_actions']}")
    print(f"Training Time:  {train_metrics['training_time']:.2f}s")


def run_time_limited_mode(args):
    if not args.agent:
        print("Error: --agent is required for time-limited mode")
        sys.exit(1)
    if not args.time:
        print("Error: --time is required for time-limited mode")
        sys.exit(1)

    AgentClass = AGENT_REGISTRY[args.agent]
    params = OPTIMIZED_PARAMS.get(args.agent, {})
    agent = AgentClass(**params)

    print(f"\nTraining {agent.name} with time limit of {args.time}s...")
    print(f"Optimized Parameters: {agent.get_params()}\n")
    train_metrics = train(agent, time_limit=args.time)

    print(f"\nCompleted {train_metrics['episodes']} episodes in {train_metrics['training_time']:.2f}s")
    print(f"\nTesting {agent.name} for {args.test} episodes...")
    test_metrics = evaluate(agent, n_episodes=args.test,
                            display_episodes=args.display)

    print(f"\n{'='*40}")
    print(f"Results for {agent.name}")
    print(f"{'='*40}")
    print(f"Mean Reward:    {test_metrics['mean_reward']:.2f}")
    print(f"Mean Steps:     {test_metrics['mean_steps']:.1f}")
    print(f"Success Rate:   {test_metrics['success_rate']*100:.1f}%")
    print(f"Illegal Actions: {test_metrics['illegal_actions']}")
    print(f"Training Time:  {train_metrics['training_time']:.2f}s")
    print(f"Episodes Trained: {train_metrics['episodes']}")


def run_benchmark_mode(args):
    if not args.train:
        print("Error: --train is required for benchmark mode")
        sys.exit(1)

    print(f"\nRunning benchmark: {args.train} train / {args.test} test episodes\n")
    results = run_benchmark(train_episodes=args.train, test_episodes=args.test)

    print(f"\n{'='*60}")
    print(f"{'Agent':<15} {'Reward':>10} {'Steps':>10} {'Success':>10} {'Time':>10}")
    print(f"{'='*60}")
    for name, data in results.items():
        t = data["test"]
        tr = data["train"]
        print(f"{data['agent_name']:<15} {t['mean_reward']:>10.2f} "
              f"{t['mean_steps']:>10.1f} {t['success_rate']*100:>9.1f}% "
              f"{tr['training_time']:>9.2f}s")

    report_path = generate_report(results)
    print(f"\nPDF report saved to: {report_path}")


def main():
    args = parse_args()

    if args.mode == "user":
        run_user_mode(args)
    elif args.mode == "time-limited":
        run_time_limited_mode(args)
    elif args.mode == "benchmark":
        run_benchmark_mode(args)


if __name__ == "__main__":
    main()
