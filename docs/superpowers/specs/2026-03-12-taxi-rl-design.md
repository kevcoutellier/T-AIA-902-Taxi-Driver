# Taxi-v3 Reinforcement Learning Agent — Design Spec

## Goal

Build a Python program that solves the Gymnasium Taxi-v3 environment using multiple model-free RL algorithms, benchmarks them against a brute-force baseline, and produces a PDF report with metrics, plots, and commentary.

## Architecture

```
taxidirver/
├── main.py              # Entry point, CLI argument parsing
├── env_wrapper.py       # Gymnasium Taxi-v3 environment wrapper
├── agents/
│   ├── base.py          # Abstract base agent interface
│   ├── bruteforce.py    # Random/brute-force baseline agent
│   ├── q_learning.py    # Q-Learning agent
│   ├── sarsa.py         # SARSA agent
│   └── dqn.py           # Deep Q-Learning agent (PyTorch)
├── training.py          # Training loop (shared across agents)
├── evaluation.py        # Testing loop, episode display, metrics collection
├── benchmark.py         # Run all agents, collect comparison data
├── report.py            # Generate PDF report with plots (matplotlib + FPDF)
└── requirements.txt
```

## Agent Interface

All agents implement a common interface via `BaseAgent`:

- `select_action(state, info=None)` — returns action given current state
- `learn(state, action, reward, next_state, done, info=None)` — update from a transition
- `get_params()` — return dict of current hyperparameters
- `name` property — human-readable agent name

## Algorithms

### 1. Brute-Force (Random Baseline)
- Selects random valid actions each step
- Uses `info["action_mask"]` when available to avoid illegal moves
- Expected performance: ~350 steps per episode

### 2. Q-Learning (Off-Policy)
- Tabular Q-table (500 states x 6 actions)
- Update rule: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
- Epsilon-greedy exploration with decay
- Default params: alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01

### 3. SARSA (On-Policy)
- Tabular, same structure as Q-Learning
- Update rule: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a)) where a' is the next action actually taken
- Same epsilon-greedy strategy
- Default params: alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01

### 4. Deep Q-Network (DQN)
- PyTorch feedforward network: input(500 one-hot) -> 128 -> 128 -> output(6)
- Experience replay buffer (capacity 10000)
- Target network updated every 100 steps
- Adam optimizer, lr=0.001
- Batch size 64, gamma=0.99
- Epsilon-greedy with decay

## Modes

### User Mode (`--mode user`)
CLI flags for all hyperparameters:
- `--alpha` (learning rate)
- `--gamma` (discount factor)
- `--epsilon` (initial exploration)
- `--epsilon-decay`
- `--epsilon-min`
- `--agent` (q_learning, sarsa, dqn, bruteforce)
- `--train` (number of training episodes)
- `--test` (number of testing episodes)

### Time-Limited Mode (`--mode time-limited`)
- `--time` (seconds allowed for training)
- Uses pre-tuned optimal parameters per agent
- Trains as many episodes as fit within the time budget
- `--agent` and `--test` still required

## Output

### Terminal Output
- Training progress (episode number, running reward average every 100 episodes)
- After testing: mean steps, mean reward, success rate
- Display of random episodes during testing (rendered via `env.render()` in ANSI mode)

### Benchmark Mode (`--mode benchmark`)
- Runs all 4 agents with same train/test counts
- Prints comparison table to terminal
- Generates plots and PDF report

## Metrics Collected
- Reward per episode (training + testing)
- Steps per episode (training + testing)
- Success rate (passenger delivered)
- Convergence speed (episodes to reach mean reward > 7)
- Total training time
- Illegal action count

## PDF Report (report.py)

Generated via matplotlib (plots) + FPDF2 (PDF assembly):

1. **Introduction** — problem description, environment details
2. **Algorithm descriptions** — brief explanation of each
3. **Benchmark results table** — all agents side by side
4. **Plots:**
   - Reward curve per agent (training)
   - Steps per episode per agent (training)
   - Comparison bar charts (mean reward, mean steps, success rate)
   - Convergence comparison
5. **Commentary** — justification of algorithm choices, parameter tuning strategy
6. **Optimization strategy** — parameter search description, reward shaping

## Dependencies

- gymnasium (Taxi-v3 environment)
- numpy
- matplotlib
- torch (PyTorch for DQN)
- fpdf2 (PDF generation)

## CLI Examples

```bash
# User mode with Q-Learning
python main.py --mode user --agent q_learning --train 5000 --test 100 --alpha 0.1 --gamma 0.99

# Time-limited mode
python main.py --mode time-limited --agent q_learning --time 30 --test 100

# Benchmark all agents
python main.py --mode benchmark --train 5000 --test 100
```
