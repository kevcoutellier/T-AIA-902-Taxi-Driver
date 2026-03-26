"""
main.py — Point d'entrée du programme Taxi Driver.

Deux modes :
1. USER MODE : l'utilisateur choisit les hyperparamètres
   -> python main.py user --alpha 0.1 --gamma 0.99 --train 5000 --test 100

2. TIME-LIMITED MODE : paramètres pré-optimisés, entraîne dans un temps donné
   -> python main.py time --time 30 --test 100

3. BENCHMARK MODE : lance le grid search complet + génère les graphiques
   -> python main.py benchmark
"""

import argparse
import time
import numpy as np
from bruteforce import run_bruteforce, display_bruteforce_episode
from qlearning import train_qlearning, test_qlearning, display_qlearning_episode
from benchmark import (
    plot_training_curves, plot_comparison,
    run_grid_search, plot_grid_search, print_results_table
)


# Paramètres optimisés (déterminés après le grid search)
# Ces valeurs seront mises à jour au fur et à mesure du benchmark
OPTIMIZED_PARAMS = {
    "alpha": 0.5,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.999,
}


def mode_user(args):
    """
    Mode utilisateur : l'utilisateur entre ses propres paramètres.

    Séquence :
    1. Brute-force (baseline)
    2. Q-Learning avec les paramètres choisis
    3. Comparaison
    4. Affichage d'épisodes
    """
    print("\n" + "=" * 60)
    print("  MODE UTILISATEUR — Paramètres personnalisés")
    print("=" * 60)
    print(f"  alpha         : {args.alpha}")
    print(f"  gamma         : {args.gamma}")
    print(f"  epsilon       : {args.epsilon}")
    print(f"  epsilon_decay : {args.epsilon_decay}")
    print(f"  train episodes: {args.train}")
    print(f"  test episodes : {args.test}")
    print("=" * 60)

    # --- Étape 1 : Baseline brute-force ---
    print("\n[1/4] Brute-force baseline...")
    bf_results = run_bruteforce(args.test)

    # --- Étape 2 : Entraînement Q-Learning ---
    print("\n[2/4] Entraînement Q-Learning...")
    agent, history = train_qlearning(
        n_episodes=args.train,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    # --- Étape 3 : Test Q-Learning ---
    print("\n[3/4] Test Q-Learning...")
    ql_results = test_qlearning(agent, args.test)

    # --- Résultats ---
    print_results_table("Comparaison des résultats", [
        ("Brute-Force", bf_results),
        ("Q-Learning", ql_results),
    ])

    # --- Graphiques ---
    plot_training_curves(history, title="Q-Learning Training (User Mode)")
    plot_comparison(bf_results, ql_results)

    # --- Affichage d'épisodes ---
    print("\n[4/4] Affichage d'épisodes aléatoires...")
    display_bruteforce_episode()
    display_qlearning_episode(agent)


def mode_time_limited(args):
    """
    Mode temps limité : entraîne avec les meilleurs paramètres
    pendant un temps donné.

    L'idée : on ne sait pas combien d'épisodes faire,
    mais on a un budget de temps. Le programme entraîne
    le plus possible dans ce temps.
    """
    print("\n" + "=" * 60)
    print(f"  MODE TEMPS LIMITÉ — {args.time} secondes")
    print("=" * 60)

    params = OPTIMIZED_PARAMS.copy()
    time_limit = args.time

    # On entraîne par batchs de 100 épisodes tant qu'on a du temps
    from environment import create_env, get_env_info
    from qlearning import QLearningAgent

    env = create_env(render_mode=None)
    n_states, n_actions = get_env_info(env)
    agent = QLearningAgent(n_states, n_actions, **params)

    total_episodes = 0
    all_rewards = []
    all_steps = []

    start_time = time.time()

    print(f"Entraînement en cours (max {time_limit}s)...")

    while (time.time() - start_time) < time_limit:
        # Jouer un épisode
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            steps += 1
            state = next_state
            if terminated or truncated:
                break

        agent.decay_epsilon()
        all_rewards.append(total_reward)
        all_steps.append(steps)
        total_episodes += 1

    elapsed = time.time() - start_time
    env.close()

    print(f"\nEntraînement terminé : {total_episodes} épisodes en {elapsed:.1f}s")

    # Test
    print(f"\nTest sur {args.test} épisodes...")
    ql_results = test_qlearning(agent, args.test)

    print(f"\n  Mean steps  : {ql_results['mean_steps']:.1f}")
    print(f"  Mean reward : {ql_results['mean_reward']:.1f}")

    # Affichage d'un épisode
    display_qlearning_episode(agent)


def mode_benchmark(args):
    """
    Mode benchmark : lance le grid search complet.

    Séquence :
    1. Brute-force baseline
    2. Q-Learning avec paramètres par défaut (1er résultat non optimisé)
    3. Grid search sur alpha
    4. Grid search sur gamma
    5. Grid search sur epsilon_decay
    6. Entraînement final avec les meilleurs paramètres
    7. Comparaison globale + graphiques
    """
    train_ep = args.train
    test_ep = args.test

    print("\n" + "=" * 60)
    print("  MODE BENCHMARK — Analyse complète")
    print(f"  Training: {train_ep} episodes | Test: {test_ep} episodes")
    print("=" * 60)

    # --- 1. Brute-force ---
    print("\n[1/7] Brute-force baseline...")
    bf_results = run_bruteforce(test_ep)
    print(f"   -> Mean steps: {bf_results['mean_steps']:.1f} | "
          f"Mean reward: {bf_results['mean_reward']:.1f}")

    # --- 2. Q-Learning non optimisé ---
    print("\n[2/7] Q-Learning NON optimisé (paramètres par défaut)...")
    agent_naive, history_naive = train_qlearning(
        train_ep, alpha=0.1, gamma=0.9, epsilon=1.0,
        epsilon_min=0.05, epsilon_decay=0.99
    )
    ql_naive_results = test_qlearning(agent_naive, test_ep)
    plot_training_curves(history_naive, title="Q-Learning NON optimisé",
                         filename="training_naive.png")
    print(f"   -> Mean steps: {ql_naive_results['mean_steps']:.1f} | "
          f"Mean reward: {ql_naive_results['mean_reward']:.1f}")

    # --- 3. Grid search : alpha ---
    print("\n[3/7] Grid search sur ALPHA...")
    base_params = {"gamma": 0.99, "epsilon": 1.0, "epsilon_min": 0.01,
                   "epsilon_decay": 0.995}
    alpha_results = run_grid_search(
        "alpha", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        base_params, train_episodes=train_ep, test_episodes=test_ep
    )
    plot_grid_search("alpha", alpha_results)
    best_alpha = min(alpha_results, key=lambda r: r["mean_steps"])["value"]
    print(f"   -> Meilleur alpha: {best_alpha}")

    # --- 4. Grid search : gamma ---
    print("\n[4/7] Grid search sur GAMMA...")
    base_params = {"alpha": best_alpha, "epsilon": 1.0, "epsilon_min": 0.01,
                   "epsilon_decay": 0.995}
    gamma_results = run_grid_search(
        "gamma", [0.5, 0.7, 0.85, 0.95, 0.99],
        base_params, train_episodes=train_ep, test_episodes=test_ep
    )
    plot_grid_search("gamma", gamma_results)
    best_gamma = min(gamma_results, key=lambda r: r["mean_steps"])["value"]
    print(f"   -> Meilleur gamma: {best_gamma}")

    # --- 5. Grid search : epsilon_decay ---
    print("\n[5/7] Grid search sur EPSILON_DECAY...")
    base_params = {"alpha": best_alpha, "gamma": best_gamma, "epsilon": 1.0,
                   "epsilon_min": 0.01}
    decay_results = run_grid_search(
        "epsilon_decay", [0.99, 0.993, 0.995, 0.997, 0.999],
        base_params, train_episodes=train_ep, test_episodes=test_ep
    )
    plot_grid_search("epsilon_decay", decay_results)
    best_decay = min(decay_results, key=lambda r: r["mean_steps"])["value"]
    print(f"   -> Meilleur epsilon_decay: {best_decay}")

    # --- 6. Entraînement final optimisé ---
    print("\n[6/7] Entraînement OPTIMISÉ final...")
    agent_opt, history_opt = train_qlearning(
        train_ep,
        alpha=best_alpha, gamma=best_gamma,
        epsilon=1.0, epsilon_min=0.01, epsilon_decay=best_decay
    )
    ql_opt_results = test_qlearning(agent_opt, test_ep)
    plot_training_curves(history_opt, title="Q-Learning OPTIMISÉ",
                         filename="training_optimized.png")

    # --- 7. Comparaison globale ---
    print("\n[7/7] Comparaison globale...")
    print_results_table("BENCHMARK COMPLET", [
        ("Brute-Force", bf_results),
        ("Q-Learning (naïf)", ql_naive_results),
        ("Q-Learning (optimisé)", ql_opt_results),
    ])

    plot_comparison(bf_results, ql_opt_results, filename="comparison_final.png")

    # Résumé des meilleurs paramètres
    print("\n" + "=" * 60)
    print("  PARAMÈTRES OPTIMAUX TROUVÉS")
    print("=" * 60)
    print(f"  alpha         : {best_alpha}")
    print(f"  gamma         : {best_gamma}")
    print(f"  epsilon_decay : {best_decay}")
    print(f"  -> Mean steps  : {ql_opt_results['mean_steps']:.1f}")
    print(f"  -> Mean reward : {ql_opt_results['mean_reward']:.1f}")
    print("=" * 60)

    # Affichage d'épisodes
    display_qlearning_episode(agent_opt)


def main():
    """
    Parse les arguments en ligne de commande et lance le mode approprié.

    Usage :
        python main.py user --alpha 0.1 --gamma 0.99 --train 5000 --test 100
        python main.py time --time 30 --test 100
        python main.py benchmark --train 5000 --test 100
    """
    parser = argparse.ArgumentParser(
        description="Taxi Driver — Reinforcement Learning on Taxi-v3"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode d'exécution")

    # --- Mode User ---
    user_parser = subparsers.add_parser("user", help="Paramètres personnalisés")
    user_parser.add_argument("--alpha", type=float, default=0.1,
                             help="Learning rate (default: 0.1)")
    user_parser.add_argument("--gamma", type=float, default=0.99,
                             help="Discount factor (default: 0.99)")
    user_parser.add_argument("--epsilon", type=float, default=1.0,
                             help="Exploration initiale (default: 1.0)")
    user_parser.add_argument("--epsilon-min", type=float, default=0.01,
                             help="Exploration minimale (default: 0.01)")
    user_parser.add_argument("--epsilon-decay", type=float, default=0.995,
                             help="Vitesse de décroissance epsilon (default: 0.995)")
    user_parser.add_argument("--train", type=int, required=True,
                             help="Nombre d'épisodes d'entraînement")
    user_parser.add_argument("--test", type=int, required=True,
                             help="Nombre d'épisodes de test")

    # --- Mode Time-limited ---
    time_parser = subparsers.add_parser("time", help="Entraînement limité en temps")
    time_parser.add_argument("--time", type=int, required=True,
                             help="Temps max d'entraînement en secondes")
    time_parser.add_argument("--test", type=int, required=True,
                             help="Nombre d'épisodes de test")

    # --- Mode Benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark complet")
    bench_parser.add_argument("--train", type=int, default=5000,
                              help="Épisodes d'entraînement par config (default: 5000)")
    bench_parser.add_argument("--test", type=int, default=100,
                              help="Épisodes de test par config (default: 100)")

    args = parser.parse_args()

    if args.mode == "user":
        mode_user(args)
    elif args.mode == "time":
        mode_time_limited(args)
    elif args.mode == "benchmark":
        mode_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
