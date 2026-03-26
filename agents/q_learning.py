import numpy as np
from .base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Agent Q-Learning - Algorithme OFF-POLICY

    Type: OFF-POLICY

    Principe:
    Q-Learning apprend la valeur optimale des paires état-action (Q-values) en utilisant
    la mise à jour de Bellman. Il est OFF-POLICY car il apprend la politique optimale
    (greedy) tout en suivant une politique d'exploration (epsilon-greedy).

    Formule de mise à jour:
    Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

    Hyperparamètres:
    - alpha (α): Taux d'apprentissage [0-1]. Contrôle à quel point les nouvelles
      informations remplacent les anciennes. Plus élevé = apprentissage plus rapide
      mais moins stable.

    - gamma (γ): Facteur d'actualisation [0-1]. Importance des récompenses futures.
      Plus élevé = l'agent planifie à plus long terme.

    - epsilon (ε): Taux d'exploration [0-1]. Probabilité de choisir une action aléatoire.
      Plus élevé = plus d'exploration, moins élevé = plus d'exploitation.

    - epsilon_decay: Décroissance d'epsilon après chaque épisode [0-1].
      Permet de réduire progressivement l'exploration.

    - epsilon_min: Valeur minimale d'epsilon [0-1].
      Garde toujours un peu d'exploration.

    Avantages:
    - Simple et efficace
    - Converge vers la politique optimale
    - Fonctionne bien sur des problèmes discrets

    Inconvénients:
    - Nécessite une table Q de taille n_states × n_actions (mémoire)
    - Ne fonctionne pas sur des espaces d'états continus
    """

    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(n_actions, n_states)

        # Hyperparamètres
        self.alpha = alpha              # Taux d'apprentissage
        self.gamma = gamma              # Facteur d'actualisation
        self.epsilon = epsilon          # Taux d'exploration
        self.epsilon_decay = epsilon_decay  # Décroissance d'epsilon
        self.epsilon_min = epsilon_min  # Epsilon minimum

        # Table Q: Q[état, action] = valeur
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state, training=True):
        """
        Sélectionne une action avec la stratégie epsilon-greedy.

        Avec probabilité epsilon: action aléatoire (exploration)
        Avec probabilité 1-epsilon: meilleure action selon Q-table (exploitation)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: meilleure action
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Mise à jour Q-Learning (OFF-POLICY).

        La clé: on utilise max(Q(s',a')) même si ce n'est pas l'action qu'on prendrait
        avec notre politique actuelle (epsilon-greedy). C'est ça qui rend Q-Learning OFF-POLICY.
        """
        # Valeur actuelle
        current_q = self.q_table[state, action]

        # Valeur maximale du prochain état (politique optimale, greedy)
        if done:
            max_next_q = 0  # Pas de futur si l'épisode est terminé
        else:
            max_next_q = np.max(self.q_table[next_state])

        # Mise à jour Q-Learning: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

        # Décroissance d'epsilon (réduit l'exploration au fil du temps)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_hyperparameters(self):
        """Retourne les hyperparamètres actuels."""
        return {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }

    def set_hyperparameters(self, **kwargs):
        """Met à jour les hyperparamètres."""
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        if 'epsilon_decay' in kwargs:
            self.epsilon_decay = kwargs['epsilon_decay']
        if 'epsilon_min' in kwargs:
            self.epsilon_min = kwargs['epsilon_min']

    def get_policy_type(self):
        """Q-Learning est OFF-POLICY."""
        return "off-policy"

    def save(self, filepath):
        """Sauvegarde la Q-table."""
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """Charge la Q-table."""
        self.q_table = np.load(filepath)
