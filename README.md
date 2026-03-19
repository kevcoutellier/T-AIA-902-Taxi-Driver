# Taxi-v3 Reinforcement Learning Solver

Solves the [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment using multiple RL agents: Brute Force, Q-Learning, SARSA, and DQN.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Three modes are available: **user**, **time-limited**, and **benchmark**.

### User Mode

Train and test a specific agent with custom parameters.

```bash
python main.py --mode user --agent q_learning --train 10000 --test 100
python main.py --mode user --agent sarsa --train 10000 --test 100
python main.py --mode user --agent dqn --train 5000 --test 100
python main.py --mode user --agent bruteforce --train 1000 --test 100
```

Optional hyperparameters: `--alpha`, `--gamma`, `--epsilon`, `--epsilon-decay`, `--epsilon-min`, `--lr`, `--batch-size`.

### Time-Limited Mode

Train an agent for a fixed duration using optimized hyperparameters.

```bash
python main.py --mode time-limited --agent q_learning --time 30 --test 100
```

### Benchmark Mode

Compare all agents side-by-side and generate a PDF report.

```bash
python main.py --mode benchmark --train 10000 --test 100
```

## Agents

| Agent | Description |
|-------|-------------|
| `bruteforce` | Random action selection (baseline) |
| `q_learning` | Tabular Q-Learning with epsilon-greedy exploration |
| `sarsa` | On-policy SARSA with epsilon-greedy exploration |
| `dqn` | Deep Q-Network with experience replay and target network |

## Tests

```bash
pytest
```
