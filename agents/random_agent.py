import numpy as np
from .base import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent aléatoire (Baseline) - Sert de référence pour comparer les performances.
    Cet agent choisit des actions complètement au hasard sans apprentissage.

    Type: Aucun (pas d'apprentissage)

    Utilité: Établir une baseline de performance. Un bon algorithme RL doit
    largement surpasser cet agent.
    """

    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)

    def select_action(self, state, training=True):
        """
        Sélectionne une action complètement aléatoire.
        """
        return np.random.randint(0, self.n_actions)

    def update(self, state, action, reward, next_state, done):
        """
        L'agent aléatoire n'apprend pas, donc cette méthode ne fait rien.
        """
        pass

    def get_hyperparameters(self):
        """
        L'agent aléatoire n'a pas d'hyperparamètres.
        """
        return {}

    def set_hyperparameters(self, **kwargs):
        """
        L'agent aléatoire n'a pas d'hyperparamètres à définir.
        """
        pass

    def get_policy_type(self):
        """
        L'agent aléatoire n'a pas de politique d'apprentissage.
        """
        return "none"

    def save(self, filepath):
        """Rien à sauvegarder pour l'agent aléatoire."""
        pass

    def load(self, filepath):
        """Rien à charger pour l'agent aléatoire."""
        pass
