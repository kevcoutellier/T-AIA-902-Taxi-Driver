import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .base import BaseAgent


class QNetwork(nn.Module):
    """
    Réseau de neurones pour approximer la fonction Q.

    Architecture:
    - Input: état (encodé en one-hot, taille = n_states)
    - Hidden layers: 2 couches cachées de 128 neurones chacune
    - Output: Q-values pour chaque action (taille = n_actions)
    """

    def __init__(self, n_states, n_actions, hidden_size=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent(BaseAgent):
    """
    Agent Deep Q-Network (DQN) - Algorithme OFF-POLICY avec réseau de neurones

    Type: OFF-POLICY

    Principe:
    DQN utilise un réseau de neurones pour approximer la fonction Q au lieu d'une table.
    Cela permet de gérer des espaces d'états très grands ou continus.
    Il utilise deux innovations clés:
    1. Experience Replay: stocke les transitions et apprend par mini-batches
    2. Target Network: réseau séparé pour stabiliser l'apprentissage

    Innovations:
    - Experience Replay Buffer: mémorise les transitions (s,a,r,s') et les rejoue
      aléatoirement pour casser les corrélations temporelles et réutiliser l'expérience.

    - Target Network: un second réseau mis à jour moins fréquemment pour calculer
      les Q-values cibles, ce qui stabilise l'apprentissage.

    Hyperparamètres:
    - alpha (learning_rate): Taux d'apprentissage [0.001-0.0001]. Contrôle la vitesse
      d'apprentissage du réseau de neurones. Plus élevé = apprentissage rapide mais instable.

    - gamma (γ): Facteur d'actualisation [0-1]. Importance des récompenses futures.

    - epsilon (ε): Taux d'exploration [0-1]. Probabilité d'action aléatoire.

    - epsilon_decay: Décroissance d'epsilon [0-1]. Réduit l'exploration progressivement.

    - epsilon_min: Epsilon minimum [0-1].

    - batch_size: Taille du mini-batch [16-128]. Nombre de transitions utilisées
      par mise à jour. Plus grand = apprentissage plus stable mais plus lent.

    - buffer_size: Taille du replay buffer [1000-100000]. Nombre de transitions
      mémorisées. Plus grand = plus de diversité mais plus de mémoire.

    - target_update: Fréquence de mise à jour du target network [100-1000].
      Nombre d'étapes entre chaque synchronisation. Plus élevé = plus stable.

    Avantages:
    - Fonctionne sur de grands espaces d'états
    - Peut généraliser à des états non vus
    - Plus puissant que les méthodes tabulaires

    Inconvénients:
    - Plus complexe et plus lent à entraîner
    - Nécessite plus de données (épisodes)
    - Hyperparamètres sensibles
    - Besoin de GPU pour de gros problèmes

    Quand utiliser DQN?
    - Espaces d'états très grands (> 10000 états)
    - États continus ou images
    - Quand on a du temps et des ressources computationnelles
    """

    def __init__(self, n_actions, n_states, alpha=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=32, buffer_size=10000, target_update=100,
                 hidden_size=128):
        super().__init__(n_actions, n_states)

        # Hyperparamètres
        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Facteur d'actualisation
        self.epsilon = epsilon          # Taux d'exploration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size    # Taille du batch
        self.buffer_size = buffer_size  # Taille du replay buffer
        self.target_update = target_update  # Fréquence de mise à jour du target network

        # Replay buffer: stocke les transitions (s, a, r, s', done)
        self.memory = deque(maxlen=buffer_size)

        # Réseaux de neurones
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        self.target_network = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        # Compteur pour la mise à jour du target network
        self.update_counter = 0

    def _state_to_tensor(self, state):
        """
        Convertit un état (entier) en tensor one-hot.
        """
        one_hot = torch.zeros(self.n_states, device=self.device)
        one_hot[state] = 1.0
        return one_hot

    def select_action(self, state, training=True):
        """
        Sélectionne une action avec epsilon-greedy.
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: meilleure action selon le réseau
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """
        Stocke la transition dans le replay buffer et entraîne le réseau.
        """
        # Stocke la transition dans le replay buffer
        self.memory.append((state, action, reward, next_state, done))

        # On n'entraîne que si on a assez de transitions
        if len(self.memory) < self.batch_size:
            return

        # Sample un mini-batch aléatoire du replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertit en tensors
        states_tensor = torch.stack([self._state_to_tensor(s) for s in states])
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.stack([self._state_to_tensor(s) for s in next_states])
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q-values actuelles: Q(s, a)
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()

        # Q-values cibles: r + γ * max(Q_target(s', a'))
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values

        # Calcul de la loss et backpropagation
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour du target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

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
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'target_update': self.target_update
        }

    def set_hyperparameters(self, **kwargs):
        """Met à jour les hyperparamètres."""
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
            # Met à jour le learning rate de l'optimiseur
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.alpha
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        if 'epsilon_decay' in kwargs:
            self.epsilon_decay = kwargs['epsilon_decay']
        if 'epsilon_min' in kwargs:
            self.epsilon_min = kwargs['epsilon_min']
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'target_update' in kwargs:
            self.target_update = kwargs['target_update']

    def get_policy_type(self):
        """DQN est OFF-POLICY."""
        return "off-policy"

    def save(self, filepath):
        """Sauvegarde le modèle."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Charge le modèle."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
