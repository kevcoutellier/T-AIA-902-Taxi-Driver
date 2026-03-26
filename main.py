"""
Application interactive de Reinforcement Learning pour le jeu Taxi-v3

Cette application permet de:
- Tester différents algorithmes RL (Random, Q-Learning, SARSA, DQN)
- Ajuster les hyperparamètres en temps réel
- Visualiser l'apprentissage sur une grille 5x5
- Obtenir des recommandations intelligentes d'hyperparamètres
- Comparer les performances on-policy vs off-policy
"""

import sys
import tkinter as tk
from ui.main_window import TaxiRLApp


def main():
    """Point d'entrée de l'application."""
    root = tk.Tk()
    app = TaxiRLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
