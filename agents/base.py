from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Classe de base pour tous les agents de Reinforcement Learning.
    """

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.n_states = n_states
        self.training_stats = {
            'rewards': [],
            'steps': [],
            'success_rate': []
        }

    @abstractmethod
    def select_action(self, state, training=True):
        """
        Sélectionne une action basée sur l'état actuel.

        Args:
            state: L'état actuel
            training: Si True, utilise l'exploration (epsilon-greedy)

        Returns:
            L'action sélectionnée
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Met à jour l'agent après une transition.

        Args:
            state: L'état actuel
            action: L'action prise
            reward: La récompense reçue
            next_state: Le nouvel état
            done: Si l'épisode est terminé
        """
        pass

    @abstractmethod
    def get_hyperparameters(self):
        """
        Retourne un dictionnaire des hyperparamètres actuels.
        """
        pass

    @abstractmethod
    def set_hyperparameters(self, **kwargs):
        """
        Met à jour les hyperparamètres de l'agent.
        """
        pass

    @abstractmethod
    def get_policy_type(self):
        """
        Retourne le type de politique: 'on-policy' ou 'off-policy'
        """
        pass

    def reset_stats(self):
        """Réinitialise les statistiques d'entraînement."""
        self.training_stats = {
            'rewards': [],
            'steps': [],
            'success_rate': []
        }

    def add_episode_stats(self, total_reward, steps, success):
        """Ajoute les statistiques d'un épisode."""
        self.training_stats['rewards'].append(total_reward)
        self.training_stats['steps'].append(steps)
        self.training_stats['success_rate'].append(1 if success else 0)
