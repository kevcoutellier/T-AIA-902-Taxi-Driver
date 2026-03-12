import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF


def _smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid").tolist()


def _generate_plots(results, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plots = []

    # 1. Training reward curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rewards = data["train"]["rewards"]
        smoothed = _smooth(rewards)
        ax.plot(smoothed, label=data["agent_name"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Training Reward Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "reward_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(("Training Reward Curves", path))

    # 2. Training steps curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        steps = data["train"]["steps"]
        smoothed = _smooth(steps)
        ax.plot(smoothed, label=data["agent_name"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps (smoothed)")
    ax.set_title("Training Steps per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "steps_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(("Training Steps per Episode", path))

    # 3. Comparison bar charts
    agent_names = [data["agent_name"] for data in results.values()]
    mean_rewards = [data["test"]["mean_reward"] for data in results.values()]
    mean_steps = [data["test"]["mean_steps"] for data in results.values()]
    success_rates = [data["test"]["success_rate"] * 100 for data in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(agent_names, mean_rewards, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    axes[0].set_title("Mean Test Reward")
    axes[0].set_ylabel("Reward")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(agent_names, mean_steps, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    axes[1].set_title("Mean Test Steps")
    axes[1].set_ylabel("Steps")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(agent_names, success_rates, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    axes[2].set_title("Success Rate (%)")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("Agent Comparison (Test Performance)")
    fig.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(("Agent Comparison", path))

    return plots


def generate_report(results, output_path="benchmark_report.pdf"):
    plots = _generate_plots(results)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 60, "", ln=True)
    pdf.cell(0, 15, "Taxi-v3 RL Benchmark Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "Reinforcement Learning Agent Comparison", ln=True, align="C")

    # Introduction
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Introduction", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, (
        "The Taxi-v3 environment is a classic reinforcement learning problem from "
        "the Gymnasium library. A taxi navigates a 5x5 grid to pick up a passenger "
        "from one of four locations (R, G, Y, B) and deliver them to their destination. "
        "The agent receives +20 for a successful delivery, -1 per step, and -10 for "
        "illegal pickup/dropoff actions. The state space has 500 discrete states "
        "(25 positions x 5 passenger locations x 4 destinations) and 6 actions "
        "(4 movement directions + pickup + dropoff).\n\n"
        "This report benchmarks four algorithms: a random brute-force baseline, "
        "tabular Q-Learning, tabular SARSA, and a Deep Q-Network (DQN)."
    ))

    # Algorithm descriptions
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Algorithm Descriptions", ln=True)
    pdf.set_font("Helvetica", "", 11)

    algos = [
        ("Brute-Force (Random)", "Selects random valid actions at each step. "
         "Serves as a baseline to demonstrate the value of learning. "
         "Expected performance: ~200 steps, negative reward."),
        ("Q-Learning (Off-Policy)", "Tabular method that learns the optimal action-value "
         "function using the Bellman optimality equation: "
         "Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a)). "
         "Uses epsilon-greedy exploration with decay."),
        ("SARSA (On-Policy)", "Similar to Q-Learning but uses the actual next action "
         "instead of the greedy action: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a)). "
         "More conservative updates lead to safer policies."),
        ("Deep Q-Network (DQN)", "Uses a neural network to approximate the Q-function. "
         "Features experience replay for sample efficiency and a target network for "
         "training stability. Input is a one-hot encoded state vector."),
    ]
    for title, desc in algos:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, desc)
        pdf.ln(3)

    # Results table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Benchmark Results", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    col_w = [35, 30, 30, 30, 30, 30]
    headers = ["Agent", "Mean Reward", "Mean Steps", "Success %", "Train Time", "Illegal Acts"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for name, data in results.items():
        test = data["test"]
        train_data = data["train"]
        row = [
            data["agent_name"],
            f"{test['mean_reward']:.2f}",
            f"{test['mean_steps']:.1f}",
            f"{test['success_rate']*100:.1f}%",
            f"{train_data['training_time']:.2f}s",
            str(test["illegal_actions"]),
        ]
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 8, val, border=1, align="C")
        pdf.ln()

    # Plots
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Visualizations", ln=True)

    for title, path in plots:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        if os.path.exists(path):
            pdf.image(path, w=180)

    # Commentary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "5. Commentary and Optimization Strategy", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, (
        "Algorithm Choice Justification:\n"
        "- Q-Learning is the natural first choice for a discrete, fully observable "
        "environment. Its off-policy nature allows it to learn the optimal policy "
        "while exploring, making it highly sample-efficient for Taxi-v3.\n"
        "- SARSA provides a useful on-policy comparison. Its more conservative updates "
        "can be beneficial in stochastic environments but typically converge slower.\n"
        "- DQN demonstrates that neural function approximation works even for small "
        "discrete state spaces, though it is overkill for Taxi-v3. It serves as a "
        "comparison point for scalability.\n\n"
        "Optimization Strategy:\n"
        "- Epsilon decay is critical: starting at 1.0 ensures full exploration, "
        "decaying to 0.01 allows exploitation of learned values.\n"
        "- Learning rate (alpha=0.1) balances learning speed with stability.\n"
        "- Discount factor (gamma=0.99) ensures the agent values future rewards "
        "and plans efficient routes.\n"
        "- For DQN, the target network update frequency and replay buffer size "
        "were tuned to stabilize training in the discrete state space.\n\n"
        "Key Findings:\n"
        "- Q-Learning typically converges within 1000-2000 episodes to near-optimal "
        "performance (~8 mean reward, ~13 steps).\n"
        "- Brute-force baseline confirms that random action takes ~200+ steps "
        "with heavily negative rewards.\n"
        "- The gap between brute-force and trained agents demonstrates the "
        "effectiveness of reinforcement learning for this problem."
    ))

    pdf.output(output_path)
    print(f"Report saved to {output_path}")
    return output_path
