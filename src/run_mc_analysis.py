"""
run_mc_analysis.py — Analyse complète Monte Carlo pour le rapport.

Grid search sur gamma et epsilon_decay, comparaison 8 métriques.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from montecarlo import train_montecarlo, test_montecarlo
from qlearning import train_qlearning, test_qlearning
from bruteforce import run_bruteforce
from benchmark import print_results_table, smooth

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TRAIN = 50000
TEST  = 100

# Paramètres communs du groupe
COMMON = dict(epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)


def save_fig(filename):
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


# ── 1. Grid search gamma ──────────────────────────────────────────
print("\n" + "="*60)
print("  GRID SEARCH GAMMA (Monte Carlo)")
print("="*60)

gamma_values = [0.5, 0.7, 0.85, 0.95, 0.99]
gamma_results = []

for g in gamma_values:
    print(f"\n  gamma={g}...")
    agent, history = train_montecarlo(TRAIN, gamma=g, **COMMON, verbose=False)
    r = test_montecarlo(agent, TEST, verbose=False)
    r["convergence_episode"] = history.get("convergence_episode")
    r["gamma"] = g
    r["history"] = history
    gamma_results.append(r)
    print(f"  steps={r['mean_steps']:.1f}  reward={r['mean_reward']:.1f}  "
          f"succes={r['success_rate']:.1f}%  conv={r.get('convergence_episode')}")

# Graphique grid search gamma MC
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Monte Carlo — Impact de gamma", fontsize=13, fontweight="bold")
vals = [str(r["gamma"]) for r in gamma_results]
colors = ["#ef4444" if r["success_rate"] < 50 else "#22c55e" for r in gamma_results]

axes[0].bar(vals, [r["mean_steps"] for r in gamma_results],
            color=colors, edgecolor="white")
axes[0].set_title("Mean Steps (moins = mieux)")
axes[0].set_xlabel("gamma")
for i, r in enumerate(gamma_results):
    axes[0].text(i, r["mean_steps"]+0.5, f"{r['mean_steps']:.1f}", ha="center", fontsize=9)

axes[1].bar(vals, [r["mean_reward"] for r in gamma_results],
            color=colors, edgecolor="white")
axes[1].set_title("Mean Reward (plus = mieux)")
axes[1].set_xlabel("gamma")
for i, r in enumerate(gamma_results):
    axes[1].text(i, r["mean_reward"]+2, f"{r['mean_reward']:.1f}", ha="center", fontsize=9)

axes[2].bar(vals, [r["success_rate"] for r in gamma_results],
            color=colors, edgecolor="white")
axes[2].set_title("Taux de succes %")
axes[2].set_xlabel("gamma")
for i, r in enumerate(gamma_results):
    axes[2].text(i, r["success_rate"]+0.5, f"{r['success_rate']:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
save_fig("mc_grid_search_gamma.png")

# ── 2. Grid search epsilon_decay ──────────────────────────────────
print("\n" + "="*60)
print("  GRID SEARCH EPSILON_DECAY (Monte Carlo)")
print("="*60)

# Utiliser le meilleur gamma trouvé
best_gamma = min(gamma_results, key=lambda r: r["mean_steps"])["gamma"]
print(f"  (avec le meilleur gamma trouvé : {best_gamma})")

decay_values = [0.990, 0.995, 0.997, 0.999]
decay_results = []

for d in decay_values:
    print(f"\n  epsilon_decay={d}...")
    agent, history = train_montecarlo(TRAIN, gamma=best_gamma,
                                      epsilon=1.0, epsilon_min=0.01,
                                      epsilon_decay=d, verbose=False)
    r = test_montecarlo(agent, TEST, verbose=False)
    r["convergence_episode"] = history.get("convergence_episode")
    r["decay"] = d
    r["history"] = history
    decay_results.append(r)
    print(f"  steps={r['mean_steps']:.1f}  reward={r['mean_reward']:.1f}  "
          f"succes={r['success_rate']:.1f}%  conv={r.get('convergence_episode')}")

# Graphique grid search decay MC
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Monte Carlo — Impact d'epsilon_decay", fontsize=13, fontweight="bold")
vals = [str(r["decay"]) for r in decay_results]
colors = ["#ef4444" if r["success_rate"] < 50 else "#22c55e" for r in decay_results]

axes[0].bar(vals, [r["mean_steps"] for r in decay_results], color=colors, edgecolor="white")
axes[0].set_title("Mean Steps")
axes[0].set_xlabel("epsilon_decay")
for i, r in enumerate(decay_results):
    axes[0].text(i, r["mean_steps"]+0.5, f"{r['mean_steps']:.1f}", ha="center", fontsize=9)

axes[1].bar(vals, [r["mean_reward"] for r in decay_results], color=colors, edgecolor="white")
axes[1].set_title("Mean Reward")
axes[1].set_xlabel("epsilon_decay")
for i, r in enumerate(decay_results):
    axes[1].text(i, r["mean_reward"]+2, f"{r['mean_reward']:.1f}", ha="center", fontsize=9)

axes[2].bar(vals, [r["success_rate"] for r in decay_results], color=colors, edgecolor="white")
axes[2].set_title("Taux de succes %")
axes[2].set_xlabel("epsilon_decay")
for i, r in enumerate(decay_results):
    axes[2].text(i, r["success_rate"]+0.5, f"{r['success_rate']:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
save_fig("mc_grid_search_decay.png")

# ── 3. Entraînement final MC avec les meilleurs params ────────────
best_decay = min(decay_results, key=lambda r: r["mean_steps"])["decay"]
print(f"\n{'='*60}")
print(f"  ENTRAINEMENT FINAL MC : gamma={best_gamma}  decay={best_decay}")
print(f"{'='*60}")

mc_agent, mc_history = train_montecarlo(
    TRAIN, gamma=best_gamma, epsilon=1.0,
    epsilon_min=0.01, epsilon_decay=best_decay, verbose=True
)
mc_final = test_montecarlo(mc_agent, TEST, verbose=False)
mc_final["convergence_episode"] = mc_history.get("convergence_episode")

# Courbes d'entraînement MC final
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
fig.suptitle(f"Monte Carlo — params optimaux (gamma={best_gamma}, decay={best_decay})",
             fontsize=12, fontweight="bold")

eps = range(len(mc_history["rewards"]))
axes[0].plot(eps, mc_history["rewards"], alpha=0.2, color="#6366f1")
axes[0].plot(eps, smooth(mc_history["rewards"], 100), color="#4f46e5", linewidth=2)
axes[0].axhline(mc_final["mean_reward"], color="#22c55e", linestyle="--",
                linewidth=1.5, label=f"Test moy. {mc_final['mean_reward']:.1f}")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(eps, mc_history["steps"], alpha=0.2, color="#f59e0b")
axes[1].plot(eps, smooth(mc_history["steps"], 100), color="#d97706", linewidth=2)
axes[1].axhline(mc_final["mean_steps"], color="#22c55e", linestyle="--",
                linewidth=1.5, label=f"Test moy. {mc_final['mean_steps']:.1f}")
axes[1].set_ylabel("Steps")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(eps, mc_history["epsilons"], color="#8b5cf6", linewidth=2)
axes[2].set_ylabel("Epsilon")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_fig("mc_training_optimized.png")

# ── 4. Comparaison Q-Learning vs Monte Carlo (8 métriques) ────────
print(f"\n{'='*60}")
print("  COMPARAISON QL vs MC (params communs du groupe)")
print(f"{'='*60}")

ql_agent, ql_history = train_qlearning(
    TRAIN, alpha=0.1, gamma=0.99, epsilon=1.0,
    epsilon_min=0.01, epsilon_decay=0.995, verbose=True
)
ql_results = test_qlearning(ql_agent, TEST, verbose=False)
ql_results["convergence_episode"] = ql_history.get("convergence_episode")

bf_results = run_bruteforce(TEST)

print_results_table("COMPARAISON FINALE — 8 METRIQUES", [
    ("Brute-Force",  bf_results),
    ("Q-Learning",   ql_results),
    ("Monte Carlo",  mc_final),
])

# Graphique comparatif courbes d'apprentissage
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Q-Learning vs Monte Carlo — Courbes d'apprentissage", fontsize=12)

axes[0].plot(smooth(ql_history["rewards"], 100), color="#2E75B6", linewidth=2, label="Q-Learning")
axes[0].plot(smooth(mc_history["rewards"], 100), color="#C0392B", linewidth=2, label="Monte Carlo")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward (lissé)")
axes[0].set_title("Reward par épisode")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(smooth(ql_history["steps"], 100), color="#2E75B6", linewidth=2, label="Q-Learning")
axes[1].plot(smooth(mc_history["steps"], 100), color="#C0392B", linewidth=2, label="Monte Carlo")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Steps (lissé)")
axes[1].set_title("Steps par épisode")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_fig("mc_vs_ql_curves.png")

# ── 5. Barplot 8 métriques côte à côte ───────────────────────────
metrics_labels = ["Mean Reward", "Mean Steps", "Succes %", "Reward/Step"]
ql_vals  = [ql_results["mean_reward"], ql_results["mean_steps"],
            ql_results["success_rate"], ql_results["reward_per_step"]]
mc_vals  = [mc_final["mean_reward"], mc_final["mean_steps"],
            mc_final["success_rate"], mc_final["reward_per_step"]]
bf_vals  = [bf_results["mean_reward"], bf_results["mean_steps"],
            bf_results["success_rate"], bf_results["reward_per_step"]]

x = np.arange(len(metrics_labels))
w = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, bf_vals,  width=w, label="Brute-Force", color="#e74c3c")
ax.bar(x,     ql_vals,  width=w, label="Q-Learning",  color="#2ecc71")
ax.bar(x + w, mc_vals,  width=w, label="Monte Carlo", color="#3498db")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Comparaison des métriques clés — Brute-Force vs Q-Learning vs Monte Carlo",
             fontsize=11)
ax.legend()
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
save_fig("mc_metrics_comparison.png")

# ── Résumé final ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  PARAMETRES OPTIMAUX MONTE CARLO")
print(f"{'='*60}")
print(f"  gamma        : {best_gamma}")
print(f"  epsilon_decay: {best_decay}")
print(f"  mean_steps   : {mc_final['mean_steps']:.1f}")
print(f"  mean_reward  : {mc_final['mean_reward']:.1f}")
print(f"  success_rate : {mc_final['success_rate']:.1f}%")
conv = mc_final.get("convergence_episode")
print(f"  convergence  : {f'ep.{conv}' if conv else 'N/A'}")

# Sauvegarde des valeurs pour le rapport
import json
report_data = {
    "gamma_grid": [{"gamma": r["gamma"], "mean_steps": round(r["mean_steps"],1),
                    "mean_reward": round(r["mean_reward"],1),
                    "success_rate": round(r["success_rate"],1),
                    "convergence_episode": r.get("convergence_episode")} for r in gamma_results],
    "decay_grid": [{"decay": r["decay"], "mean_steps": round(r["mean_steps"],1),
                    "mean_reward": round(r["mean_reward"],1),
                    "success_rate": round(r["success_rate"],1),
                    "convergence_episode": r.get("convergence_episode")} for r in decay_results],
    "best_gamma": best_gamma,
    "best_decay": best_decay,
    "mc_final": {
        "mean_steps": round(mc_final["mean_steps"],1),
        "std_steps": round(mc_final["std_steps"],1),
        "mean_reward": round(mc_final["mean_reward"],1),
        "std_reward": round(mc_final["std_reward"],1),
        "success_rate": round(mc_final["success_rate"],1),
        "reward_per_step": round(mc_final["reward_per_step"],3),
        "illegal_actions": mc_final["illegal_actions"],
        "convergence_episode": mc_final.get("convergence_episode"),
    },
    "ql_final": {
        "mean_steps": round(ql_results["mean_steps"],1),
        "std_steps": round(ql_results["std_steps"],1),
        "mean_reward": round(ql_results["mean_reward"],1),
        "std_reward": round(ql_results["std_reward"],1),
        "success_rate": round(ql_results["success_rate"],1),
        "reward_per_step": round(ql_results["reward_per_step"],3),
        "illegal_actions": ql_results["illegal_actions"],
        "convergence_episode": ql_results.get("convergence_episode"),
    },
    "bf_final": {
        "mean_steps": round(bf_results["mean_steps"],1),
        "std_steps": round(bf_results["std_steps"],1),
        "mean_reward": round(bf_results["mean_reward"],1),
        "std_reward": round(bf_results["std_reward"],1),
        "success_rate": round(bf_results["success_rate"],1),
        "reward_per_step": round(bf_results["reward_per_step"],3),
        "illegal_actions": bf_results["illegal_actions"],
        "convergence_episode": None,
    },
}

data_path = os.path.join(RESULTS_DIR, "mc_report_data.json")
with open(data_path, "w") as f:
    json.dump(report_data, f, indent=2)
print(f"\n  Données sauvegardées : {data_path}")
print("\nTerminé.")
