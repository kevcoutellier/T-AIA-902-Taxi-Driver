import numpy as np
from .base import BaseAgent


class SARSAAgent(BaseAgent):
    """
    Agent SARSA (State-Action-Reward-State-Action) - Algorithme ON-POLICY

    Type: ON-POLICY

    Principe:
    SARSA apprend la valeur des paires état-action en suivant la politique actuelle
    (epsilon-greedy). Il est ON-POLICY car il met à jour les Q-values en utilisant
    l'action réellement prise par sa politique d'exploration, pas la meilleure action.

    Formule de mise à jour:
    Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    où a' est l'action réellement choisie dans s' (pas max)

    Différence clé avec Q-Learning:
    - Q-Learning (OFF-POLICY): utilise max(Q(s',a')) - apprend la politique optimale
    - SARSA (ON-POLICY): utilise Q(s',a') où a' est l'action prise - apprend la politique suivie

    Hyperparamètres:
    - alpha (α): Taux d'apprentissage [0-1]. Contrôle la vitesse d'apprentissage.
      Plus élevé = apprentissage rapide mais instable.

    - gamma (γ): Facteur d'actualisation [0-1]. Importance des récompenses futures.
      Plus élevé = planification à long terme.

    - epsilon (ε): Taux d'exploration [0-1]. Probabilité d'action aléatoire.
      Plus élevé = plus d'exploration.

    - epsilon_decay: Décroissance d'epsilon [0-1]. Réduit l'exploration progressivement.

    - epsilon_min: Epsilon minimum [0-1]. Garde un minimum d'exploration.

    Avantages:
    - Plus prudent que Q-Learning (prend en compte l'exploration dans l'apprentissage)
    - Mieux dans les environnements risqués
    - Converge vers une politique stable

    Inconvénients:
    - Apprentissage potentiellement plus lent que Q-Learning
    - Converge vers une politique sous-optimale si epsilon ne décroît pas à 0
    - Nécessite de choisir la prochaine action avant la mise à jour

    Quand utiliser SARSA vs Q-Learning?
    - SARSA: Environnements avec pénalités importantes, où la sécurité compte
    - Q-Learning: Environnements où on veut la politique optimale rapidement
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

        # SARSA a besoin de stocker la prochaine action
        self.next_action = None

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
        Mise à jour SARSA (ON-POLICY).

        SARSA doit connaître la prochaine action a' avant de faire la mise à jour.
        On utilise Q(s', a') où a' est l'action qui SERA prise, pas la meilleure.
        """
        # Valeur actuelle
        current_q = self.q_table[state, action]

        # On doit choisir la prochaine action selon notre politique actuelle
        if done:
            next_q = 0  # Pas de futur si l'épisode est terminé
        else:
            # Choix de la prochaine action selon epsilon-greedy (ON-POLICY!)
            if np.random.random() < self.epsilon:
                next_action = np.random.randint(0, self.n_actions)
            else:
                next_action = np.argmax(self.q_table[next_state])

            next_q = self.q_table[next_state, next_action]

        # Mise à jour SARSA: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        # Note: on utilise Q(s',a') pas max(Q(s',a'))
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state, action] = new_q

        # Décroissance d'epsilon
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
        """SARSA est ON-POLICY."""
        return "on-policy"

    def save(self, filepath):
        """Sauvegarde la Q-table."""
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """Charge la Q-table."""
        self.q_table = np.load(filepath)
