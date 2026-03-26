"""
Benchmark comparatif : Random vs Q-Learning vs SARSA vs DQN
Usage :
    python benchmark.py
    python benchmark.py --map 8x8 --train 5000 --test 500
    python benchmark.py --no-slippery
"""
import argparse
import sys
import time

import numpy as np

from env.frozen_lake import FrozenLakeEnv
from agents.random_agent import RandomAgent
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent


# ------------------------------------------------------------------
# Helpers d'affichage
# ------------------------------------------------------------------

def hr(char="-", n=60):
    print(char * n)

def print_header(title: str):
    hr("=")
    print(f"  {title}")
    hr("=")

def print_section(title: str):
    hr()
    print(f"  {title}")
    hr()


# ------------------------------------------------------------------
# Benchmark principal
# ------------------------------------------------------------------

def run_benchmark(map_name: str, n_train: int, m_test: int, is_slippery: bool):
    env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)

    print_header(f"FindYourIcePath — Benchmark")
    print(f"  Map          : {map_name}  ({env.nrow}×{env.ncol})")
    print(f"  Slippery     : {is_slippery}")
    print(f"  Train eps    : {n_train}")
    print(f"  Test  eps    : {m_test}")
    print()
    print(env.grid_repr())
    print()

    agents = [
        RandomAgent(env),
        QLearningAgent(env),
        SARSAAgent(env),
        DQNAgent(env),
    ]

    results = {}

    for agent in agents:
        print_section(f"Agent : {agent.name}")

        if agent.name == "Random":
            # Pas d'entraînement
            print("  (pas d'entraînement)")
            train_res = None
        else:
            print(f"  Entraînement sur {n_train} épisodes...")
            t0 = time.time()
            train_res = agent.train(n_train)
            print(f"  ✓ terminé en {time.time() - t0:.2f}s")
            print(f"  [Train] win rate : {train_res.win_rate * 100:.1f}%  |  "
                  f"avg reward : {train_res.avg_reward:.4f}  |  "
                  f"avg steps : {train_res.avg_steps:.1f}")

        print(f"  Test sur {m_test} épisodes...")
        test_res = agent.test(m_test)
        print(f"  [Test]  win rate : {test_res.win_rate * 100:.1f}%  |  "
              f"avg reward : {test_res.avg_reward:.4f}  |  "
              f"avg steps : {test_res.avg_steps:.1f}")

        results[agent.name] = {
            "train": train_res,
            "test": test_res,
        }

    # ------------------------------------------------------------------
    # Tableau récapitulatif
    # ------------------------------------------------------------------
    print_header("Récapitulatif")
    header = f"{'Agent':<15} {'Win Rate (test)':>16} {'Avg Reward':>12} {'Avg Steps':>11} {'Train Time':>12}"
    print(header)
    hr()
    for name, res in results.items():
        tr = res["test"]
        t_time = res["train"].train_duration if res["train"] else 0.0
        print(
            f"{name:<15} {tr.win_rate * 100:>15.1f}%"
            f" {tr.avg_reward:>12.4f}"
            f" {tr.avg_steps:>11.1f}"
            f" {t_time:>11.2f}s"
        )
    hr()

    # ------------------------------------------------------------------
    # Conclusion automatique
    # ------------------------------------------------------------------
    print()
    print("  Conclusion :")
    best = max(
        [(n, r["test"].win_rate) for n, r in results.items()],
        key=lambda x: x[1],
    )
    print(f"  → Meilleur agent : {best[0]} ({best[1]*100:.1f}% de victoires)")
    random_wr = results["Random"]["test"].win_rate
    for name in ["Q-Learning", "SARSA", "DQN"]:
        if name in results:
            wr = results[name]["test"].win_rate
            gain = wr - random_wr
            print(f"  → {name} gagne {gain*100:+.1f}pp vs Random")

    print()


# ------------------------------------------------------------------
# Utilitaire : rejouer un épisode avec un agent entraîné
# ------------------------------------------------------------------

def replay_episode(agent, label: str):
    """Rejoue un épisode en affichant chaque step."""
    print_section(f"Replay : {label}")
    env = agent.env
    state = env.reset()
    print(f"  Départ : état {state} — {env.state_to_pos(state)}")
    step = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        row, col = env.state_to_pos(next_state)
        cell = env.desc[row][col]
        print(f"  Step {step+1:2d} | action={action} | état {next_state} ({row},{col}) [{cell}]"
              f" | reward={reward}")
        state = next_state
        step += 1
        if terminated or truncated:
            outcome = "GOAL !" if reward > 0 else "TROU !"
            print(f"  → {outcome} en {step} steps")
            break


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FindYourIcePath")
    p.add_argument("--map", default="4x4", choices=["4x4", "8x8"])
    p.add_argument("--train", type=int, default=2000, metavar="N")
    p.add_argument("--test",  type=int, default=200,  metavar="M")
    p.add_argument("--no-slippery", action="store_true")
    p.add_argument("--replay", action="store_true", help="Rejoue un épisode après benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        map_name=args.map,
        n_train=args.train,
        m_test=args.test,
        is_slippery=not args.no_slippery,
    )

    if args.replay:
        env = FrozenLakeEnv(map_name=args.map, is_slippery=not args.no_slippery)
        agent = QLearningAgent(env)
        agent.train(args.train)
        agent.epsilon = 0.0
        replay_episode(agent, "Q-Learning (greedy)")
