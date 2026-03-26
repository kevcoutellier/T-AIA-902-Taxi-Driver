"""
Script de test pour verifier que tous les composants fonctionnent.
"""

import gymnasium as gym
from agents import RandomAgent, QLearningAgent, SARSAAgent, DQNAgent
from utils import HyperparameterAnalyzer, MetricsCalculator

def test_environment():
    """Test l'environnement Taxi."""
    print("Test de l'environnement Taxi-v3...")
    env = gym.make('Taxi-v3')
    state, _ = env.reset()
    print(f"  Etat initial: {state}")
    print(f"  Espace d'actions: {env.action_space.n}")
    print(f"  Espace d'etats: {env.observation_space.n}")
    env.close()
    print("  [OK] Environnement OK\n")

def test_agents():
    """Test la création de tous les agents."""
    print("Test des agents RL...")
    env = gym.make('Taxi-v3')
    n_actions = env.action_space.n
    n_states = env.observation_space.n

    agents = [
        ("Random", RandomAgent(n_actions, n_states)),
        ("Q-Learning", QLearningAgent(n_actions, n_states)),
        ("SARSA", SARSAAgent(n_actions, n_states)),
        ("DQN", DQNAgent(n_actions, n_states))
    ]

    for name, agent in agents:
        print(f"  - {name}: {agent.__class__.__name__}")
        print(f"    Type: {agent.get_policy_type()}")

        # Test select_action
        state, _ = env.reset()
        action = agent.select_action(state)
        print(f"    Action selectionnee: {action}")

        # Test update
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        print(f"    [OK] {name} fonctionne")

    env.close()
    print("  [OK] Tous les agents OK\n")

def test_analyzer():
    """Test l'analyseur d'hyperparametres."""
    print("Test de l'analyseur d'hyperparametres...")
    analyzer = HyperparameterAnalyzer()

    # Recommandations
    recomm = analyzer.get_recommended_hyperparameters("QLearningAgent", "default")
    print(f"  Recommandations Q-Learning: {recomm}")
    print("  [OK] Analyseur OK\n")

def test_metrics():
    """Test le calculateur de metriques."""
    print("Test du calculateur de metriques...")
    calc = MetricsCalculator()

    stats = {
        'rewards': [1, 2, 3, 4, 5],
        'steps': [10, 15, 12, 11, 9],
        'success_rate': [1, 1, 0, 1, 1]
    }

    metrics = calc.calculate_metrics(stats)
    print(f"  Metriques calculees: avg_reward={metrics['avg_reward']:.2f}")
    print("  [OK] Calculateur OK\n")

def test_quick_episode():
    """Execute un episode rapide avec Q-Learning."""
    print("Test d'un episode rapide avec Q-Learning...")
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(env.action_space.n, env.observation_space.n)

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:
        action = agent.select_action(state, training=True)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, done or truncated)

        total_reward += reward
        steps += 1
        state = next_state

        if done or truncated:
            break

    print(f"  Episode termine: {steps} steps, recompense: {total_reward}")
    print("  [OK] Episode OK\n")

    env.close()

if __name__ == "__main__":
    print("="*60)
    print("TEST DE CONFIGURATION - TAXI DRIVER RL")
    print("="*60 + "\n")

    try:
        test_environment()
        test_agents()
        test_analyzer()
        test_metrics()
        test_quick_episode()

        print("="*60)
        print("[SUCCESS] TOUS LES TESTS REUSSIS!")
        print("="*60)
        print("\nVous pouvez maintenant lancer l'application:")
        print("  python main.py")
        print("")

    except Exception as e:
        print(f"\n[ERROR] ERREUR: {e}")
        import traceback
        traceback.print_exc()
