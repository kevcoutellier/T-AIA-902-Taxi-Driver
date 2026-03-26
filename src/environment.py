"""
environment.py — Wrapper autour de l'environnement Taxi-v3 de Gymnasium.

Taxi-v3 est une grille 5x5 avec :
- 25 positions possibles pour le taxi
- 5 positions pour le passager (R, G, Y, B, ou dans le taxi)
- 4 destinations (R, G, Y, B)
→ 500 états distincts (25 × 5 × 4)

6 actions : South(0), North(1), East(2), West(3), Pickup(4), Dropoff(5)

Rewards :
- Chaque pas      : -1
- Pickup/Dropoff illégal : -10
- Dropoff correct  : +20
"""

import gymnasium as gym


def create_env(render_mode=None):
    """
    Crée et retourne un environnement Taxi-v3.

    Args:
        render_mode: None pour l'entraînement (rapide, pas d'affichage),
                     "ansi" pour afficher la grille en texte dans le terminal,
                     "human" pour une fenêtre graphique.

    Returns:
        L'environnement Gymnasium prêt à l'emploi.
    """
    env = gym.make("Taxi-v3", render_mode=render_mode)
    return env


def get_env_info(env):
    """
    Retourne les informations clés de l'environnement.
    Utile pour initialiser la Q-table à la bonne taille.

    Returns:
        n_states  : nombre d'états (500)
        n_actions : nombre d'actions (6)
    """
    n_states = env.observation_space.n    # 500
    n_actions = env.action_space.n        # 6
    return n_states, n_actions


def run_episode(env, select_action_fn, max_steps=200):
    """
    Joue un épisode complet et retourne les métriques.

    C'est la boucle fondamentale du RL :
      1. Observer l'état
      2. Choisir une action
      3. Exécuter l'action
      4. Observer le nouvel état + reward
      5. Répéter

    Args:
        env             : l'environnement Taxi-v3
        select_action_fn: fonction(state) → action (stratégie de l'agent)
        max_steps       : limite de pas pour éviter les boucles infinies

    Returns:
        total_reward : somme des rewards de l'épisode
        steps        : nombre de pas effectués
        done         : True si l'épisode s'est terminé normalement
    """
    # Reset : on remet le jeu à zéro, on obtient l'état initial
    state, _info = env.reset()

    total_reward = 0
    steps = 0

    for step in range(max_steps):
        # L'agent choisit une action selon sa stratégie
        action = select_action_fn(state)

        # On exécute l'action dans l'environnement
        # - next_state : le nouvel état après l'action
        # - reward     : la récompense obtenue
        # - terminated : True si le passager est déposé au bon endroit
        # - truncated  : True si on a dépassé la limite de pas
        next_state, reward, terminated, truncated, _info = env.step(action)

        total_reward += reward
        steps += 1
        state = next_state

        # L'épisode est fini si le passager est déposé ou si on a timeout
        if terminated or truncated:
            break

    return total_reward, steps, terminated
