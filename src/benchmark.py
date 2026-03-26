"""
benchmark.py — Outils de benchmark et génération de graphiques.

Ce module génère les graphiques et tableaux nécessaires au rapport.
Chaque graphique répond à une question spécifique :
- Comment l'agent apprend-il au fil du temps ?
- Quel paramètre donne les meilleurs résultats ?
- À quel point l'agent RL est-il meilleur que le brute-force ?
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from qlearning import train_qlearning, test_qlearning


# Dossier où sauvegarder les graphiques
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def smooth(data, window=50):
    """
    Lisse une courbe avec une moyenne mobile.

    Les rewards bruts sont très bruités (varient beaucoup d'un épisode à l'autre).
    La moyenne mobile calcule la moyenne des N dernières valeurs,
    ce qui donne une courbe plus lisible.

    Exemple avec window=3 :
      données :  [1, 5, 3, 8, 2, 7]
      lissé :    [1, 3, 3, 5.3, 4.3, 5.7]

    Args:
        data   : liste de valeurs brutes
        window : taille de la fenêtre de lissage

    Returns:
        Liste lissée (même longueur)
    """
    smoothed = []
    for i in range(len(data)):
        # On prend les 'window' dernières valeurs (ou moins au début)
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i + 1]))
    return smoothed


def plot_training_curves(history, title="Q-Learning Training", filename="training_curves.png"):
    """
    Génère 3 graphiques d'entraînement empilés :
    1. Rewards par épisode   — montre si l'agent gagne de plus en plus
    2. Steps par épisode     — montre si l'agent est de plus en plus efficace
    3. Epsilon par épisode   — montre la transition exploration -> exploitation

    Args:
        history  : dict retourné par train_qlearning
        title    : titre du graphique
        filename : nom du fichier de sortie
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    episodes = range(len(history["rewards"]))

    # Graphique 1 : Rewards
    axes[0].plot(episodes, history["rewards"], alpha=0.3, color='blue', label='Brut')
    axes[0].plot(episodes, smooth(history["rewards"]), color='red', linewidth=2, label='Lissé (moy. mobile)')
    axes[0].set_ylabel("Reward total")
    axes[0].set_xlabel("Épisode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graphique 2 : Steps
    axes[1].plot(episodes, history["steps"], alpha=0.3, color='green', label='Brut')
    axes[1].plot(episodes, smooth(history["steps"]), color='red', linewidth=2, label='Lissé')
    axes[1].set_ylabel("Nombre de steps")
    axes[1].set_xlabel("Épisode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Graphique 3 : Epsilon
    axes[2].plot(episodes, history["epsilons"], color='purple', linewidth=2)
    axes[2].set_ylabel("Epsilon (exploration)")
    axes[2].set_xlabel("Épisode")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {filepath}")


def plot_comparison(bruteforce_results, qlearning_results, filename="comparison.png"):
    """
    Crée un barplot comparant brute-force vs Q-Learning.

    Deux métriques côte à côte :
    - Mean steps (moins = mieux)
    - Mean reward (plus = mieux)

    C'est le graphique le plus important du rapport :
    il montre visuellement l'amélioration.

    Args:
        bruteforce_results : dict retourné par run_bruteforce
        qlearning_results  : dict retourné par test_qlearning
        filename           : nom du fichier de sortie
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Brute-Force vs Q-Learning", fontsize=14, fontweight='bold')

    algos = ["Brute-Force", "Q-Learning"]

    # Steps
    steps_means = [bruteforce_results["mean_steps"], qlearning_results["mean_steps"]]
    steps_stds = [bruteforce_results["std_steps"], qlearning_results["std_steps"]]
    bars1 = axes[0].bar(algos, steps_means, yerr=steps_stds, color=['#e74c3c', '#2ecc71'], capsize=5)
    axes[0].set_ylabel("Mean Steps")
    axes[0].set_title("Nombre de pas (moins = mieux)")
    # Afficher la valeur au-dessus de chaque barre
    for bar, val in zip(bars1, steps_means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.1f}', ha='center', fontweight='bold')

    # Rewards
    reward_means = [bruteforce_results["mean_reward"], qlearning_results["mean_reward"]]
    reward_stds = [bruteforce_results["std_reward"], qlearning_results["std_reward"]]
    bars2 = axes[1].bar(algos, reward_means, yerr=reward_stds, color=['#e74c3c', '#2ecc71'], capsize=5)
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("Récompense moyenne (plus = mieux)")
    for bar, val in zip(bars2, reward_means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{val:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {filepath}")


def run_grid_search(param_name, param_values, base_params, train_episodes=5000,
                    test_episodes=100):
    """
    Fait varier UN paramètre et mesure l'impact sur les performances.

    C'est la méthode scientifique appliquée au RL :
    - On fixe tous les paramètres sauf un
    - On fait varier celui-là
    - On mesure les résultats
    - On conclut quel est le meilleur

    Exemple :
        run_grid_search("alpha", [0.01, 0.1, 0.5, 0.9], base_params)
        -> Entraîne 4 agents avec des alpha différents
        -> Retourne les performances de chacun

    Args:
        param_name     : nom du paramètre à varier ("alpha", "gamma", etc.)
        param_values   : liste de valeurs à tester
        base_params    : dict des paramètres par défaut
        train_episodes : nombre d'épisodes d'entraînement
        test_episodes  : nombre d'épisodes de test

    Returns:
        grid_results: liste de dicts, un par valeur testée
    """
    grid_results = []

    for value in param_values:
        print(f"\n--- Testing {param_name} = {value} ---")

        # On copie les paramètres de base et on modifie celui qu'on teste
        params = base_params.copy()
        params[param_name] = value

        # Entraînement
        agent, history = train_qlearning(train_episodes, **params, verbose=True)

        # Test (sans exploration)
        test_results = test_qlearning(agent, test_episodes, verbose=False)

        grid_results.append({
            "value": value,
            "mean_reward": test_results["mean_reward"],
            "mean_steps": test_results["mean_steps"],
            "std_reward": test_results["std_reward"],
            "std_steps": test_results["std_steps"],
            "history": history,
        })

        print(f"   -> Mean steps: {test_results['mean_steps']:.1f} | "
              f"Mean reward: {test_results['mean_reward']:.1f}")

    return grid_results


def plot_grid_search(param_name, grid_results, filename=None):
    """
    Génère un graphique montrant l'impact d'un paramètre sur les performances.

    Deux sous-graphiques :
    - Mean steps en fonction du paramètre
    - Mean reward en fonction du paramètre

    Args:
        param_name   : nom du paramètre ("alpha", "gamma", etc.)
        grid_results : résultats de run_grid_search
        filename     : nom du fichier (auto-généré si None)
    """
    if filename is None:
        filename = f"grid_search_{param_name}.png"

    values = [r["value"] for r in grid_results]
    mean_steps = [r["mean_steps"] for r in grid_results]
    mean_rewards = [r["mean_reward"] for r in grid_results]
    std_steps = [r["std_steps"] for r in grid_results]
    std_rewards = [r["std_reward"] for r in grid_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Impact de {param_name} sur les performances",
                 fontsize=14, fontweight='bold')

    # Steps
    axes[0].bar([str(v) for v in values], mean_steps, yerr=std_steps,
                color='#3498db', capsize=5)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Mean Steps")
    axes[0].set_title("Nombre de pas (moins = mieux)")
    for i, (v, s) in enumerate(zip(values, mean_steps)):
        axes[0].text(i, s + 1, f'{s:.1f}', ha='center', fontsize=9, fontweight='bold')

    # Rewards
    axes[1].bar([str(v) for v in values], mean_rewards, yerr=std_rewards,
                color='#e67e22', capsize=5)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("Récompense (plus = mieux)")
    for i, (v, r) in enumerate(zip(values, mean_rewards)):
        axes[1].text(i, r + 0.5, f'{r:.1f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {filepath}")


def plot_algo_comparison(results_dict, histories_dict, filename="algo_comparison.png"):
    """
    Compare plusieurs algorithmes sur un meme graphique.

    Genere 3 sous-graphiques :
    1. Barplot des performances finales (steps + reward)
    2. Courbes d'apprentissage superposees (reward par episode)
    3. Courbes de steps superposees

    Args:
        results_dict  : {"Algo1": results, "Algo2": results, ...}
        histories_dict: {"Algo1": history, "Algo2": history, ...}
        filename      : nom du fichier de sortie
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparaison inter-algorithmes", fontsize=14, fontweight='bold')

    algos = list(results_dict.keys())
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    # 1. Barplot steps
    steps_vals = [results_dict[a]["mean_steps"] for a in algos]
    steps_stds = [results_dict[a]["std_steps"] for a in algos]
    bars = axes[0, 0].bar(algos, steps_vals, yerr=steps_stds,
                          color=colors[:len(algos)], capsize=5)
    axes[0, 0].set_ylabel("Mean Steps")
    axes[0, 0].set_title("Steps (moins = mieux)")
    for bar, val in zip(bars, steps_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', fontweight='bold')

    # 2. Barplot rewards
    reward_vals = [results_dict[a]["mean_reward"] for a in algos]
    reward_stds = [results_dict[a]["std_reward"] for a in algos]
    bars = axes[0, 1].bar(algos, reward_vals, yerr=reward_stds,
                          color=colors[:len(algos)], capsize=5)
    axes[0, 1].set_ylabel("Mean Reward")
    axes[0, 1].set_title("Reward (plus = mieux)")
    for bar, val in zip(bars, reward_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', fontweight='bold')

    # 3. Courbes de reward (apprentissage)
    for i, algo in enumerate(algos):
        if algo in histories_dict and histories_dict[algo] is not None:
            data = histories_dict[algo]["rewards"]
            axes[1, 0].plot(smooth(data, 100), color=colors[i], label=algo, linewidth=2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Reward (lisse)")
    axes[1, 0].set_title("Courbes d'apprentissage - Reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Courbes de steps (apprentissage)
    for i, algo in enumerate(algos):
        if algo in histories_dict and histories_dict[algo] is not None:
            data = histories_dict[algo]["steps"]
            axes[1, 1].plot(smooth(data, 100), color=colors[i], label=algo, linewidth=2)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Steps (lisse)")
    axes[1, 1].set_title("Courbes d'apprentissage - Steps")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {filepath}")


def print_results_table(title, results_list):
    """
    Affiche un tableau formaté de résultats dans le terminal.

    Args:
        title        : titre du tableau
        results_list : liste de tuples (nom, results_dict)
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {'Algorithme':<25} {'Mean Steps':>12} {'Mean Reward':>12}")
    print(f"  {'-' * 49}")
    for name, results in results_list:
        print(f"  {name:<25} {results['mean_steps']:>12.1f} "
              f"{results['mean_reward']:>12.1f}")
    print(f"{'=' * 60}\n")
