"""
run_comparison.py — Lance la comparaison complete entre tous les algorithmes.

Ce script genere toutes les donnees et graphiques necessaires au rapport :
1. Brute-force (baseline)
2. Q-Learning optimise
3. Monte Carlo optimise
4. Comparaison croisee
"""

import numpy as np
from bruteforce import run_bruteforce
from qlearning import train_qlearning, test_qlearning
from montecarlo import train_montecarlo, test_montecarlo
from benchmark import (
    plot_training_curves, plot_comparison,
    plot_algo_comparison, print_results_table
)


def main():
    TRAIN_EPISODES = 5000
    TEST_EPISODES = 100

    print("=" * 60)
    print("  COMPARAISON COMPLETE — Brute-Force vs Q-Learning vs Monte Carlo")
    print(f"  Training: {TRAIN_EPISODES} | Test: {TEST_EPISODES}")
    print("=" * 60)

    # --- 1. Brute-force ---
    print("\n[1/5] Brute-force baseline...")
    bf_results = run_bruteforce(TEST_EPISODES)
    print(f"   -> Mean steps: {bf_results['mean_steps']:.1f} | "
          f"Mean reward: {bf_results['mean_reward']:.1f}")

    # --- 2. Q-Learning optimise ---
    print("\n[2/5] Q-Learning (optimise)...")
    ql_agent, ql_history = train_qlearning(
        TRAIN_EPISODES,
        alpha=0.5, gamma=0.95, epsilon=1.0,
        epsilon_min=0.01, epsilon_decay=0.999
    )
    ql_results = test_qlearning(ql_agent, TEST_EPISODES)
    plot_training_curves(ql_history, title="Q-Learning Optimise",
                         filename="training_qlearning_final.png")
    print(f"   -> Mean steps: {ql_results['mean_steps']:.1f} | "
          f"Mean reward: {ql_results['mean_reward']:.1f}")

    # --- 3. Monte Carlo (memes hyperparametres pour comparaison equitable) ---
    print("\n[3/5] Monte Carlo...")
    mc_agent, mc_history = train_montecarlo(
        TRAIN_EPISODES,
        gamma=0.95, epsilon=1.0,
        epsilon_min=0.01, epsilon_decay=0.999
    )
    mc_results = test_montecarlo(mc_agent, TEST_EPISODES)
    plot_training_curves(mc_history, title="Monte Carlo First-Visit",
                         filename="training_montecarlo.png")
    print(f"   -> Mean steps: {mc_results['mean_steps']:.1f} | "
          f"Mean reward: {mc_results['mean_reward']:.1f}")

    # --- 4. Tableau comparatif ---
    print("\n[4/5] Resultats...")
    print_results_table("COMPARAISON COMPLETE", [
        ("Brute-Force", bf_results),
        ("Q-Learning", ql_results),
        ("Monte Carlo", mc_results),
    ])

    # --- 5. Graphiques de comparaison ---
    print("[5/5] Generation des graphiques...")

    # Comparaison Q-Learning vs Monte Carlo (sans brute-force pour lisibilite)
    plot_algo_comparison(
        {"Q-Learning": ql_results, "Monte Carlo": mc_results},
        {"Q-Learning": ql_history, "Monte Carlo": mc_history},
        filename="algo_comparison_ql_mc.png"
    )

    # Comparaison globale incluant brute-force
    plot_algo_comparison(
        {"Brute-Force": bf_results, "Q-Learning": ql_results, "Monte Carlo": mc_results},
        {"Q-Learning": ql_history, "Monte Carlo": mc_history},
        filename="algo_comparison_all.png"
    )

    print("\nTermine ! Tous les graphiques sont dans results/")


if __name__ == "__main__":
    main()
