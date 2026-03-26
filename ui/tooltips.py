import tkinter as tk


class ToolTip:
    """
    Crée des info-bulles (tooltips) pour les widgets Tkinter.
    """

    def __init__(self, widget, text, bg_color='#FFFFE0', delay=500):
        """
        Args:
            widget: Widget auquel attacher le tooltip
            text: Texte du tooltip (peut être multi-lignes)
            bg_color: Couleur de fond
            delay: Délai en ms avant d'afficher le tooltip
        """
        self.widget = widget
        self.text = text
        self.bg_color = bg_color
        self.delay = delay
        self.tooltip_window = None
        self.schedule_id = None

        # Bind les événements
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Button>", self.on_leave)

    def on_enter(self, event=None):
        """Appelé quand la souris entre dans le widget."""
        self.schedule_id = self.widget.after(self.delay, self.show_tooltip)

    def on_leave(self, event=None):
        """Appelé quand la souris sort du widget."""
        if self.schedule_id:
            self.widget.after_cancel(self.schedule_id)
            self.schedule_id = None
        self.hide_tooltip()

    def show_tooltip(self):
        """Affiche le tooltip."""
        if self.tooltip_window:
            return

        # Position du tooltip
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25

        # Crée la fenêtre tooltip
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Contenu du tooltip
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background=self.bg_color,
            foreground="#000000",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9),
            padx=8,
            pady=6
        )
        label.pack()

    def hide_tooltip(self):
        """Cache le tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# Textes de tooltips pour les différents éléments

TOOLTIPS = {
    'agent_random': """Agent Aléatoire (Baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: Aucun apprentissage

Choisit des actions complètement au hasard.
Sert de référence pour comparer les autres
algorithmes.

Performance attendue:
• ~350-400 steps par épisode
• Récompense: -350 à -400
• Succès: très faible

Utilité: Baseline de comparaison""",

    'agent_qlearning': """Q-Learning (OFF-POLICY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: OFF-POLICY (apprend la politique
optimale indépendamment de celle suivie)

Principe:
Apprend les Q-values avec la formule:
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Utilise MAX(Q) pour la mise à jour,
donc apprend la politique optimale même
en explorant aléatoirement.

Avantages:
✓ Simple et efficace
✓ Converge vers l'optimal
✓ Apprentissage rapide

Idéal pour: Environnements standards""",

    'agent_sarsa': """SARSA (ON-POLICY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: ON-POLICY (apprend la politique
qu'il suit actuellement)

Principe:
Apprend les Q-values avec la formule:
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

Utilise l'action RÉELLE prise (a'),
donc plus prudent que Q-Learning.

Différence avec Q-Learning:
• Q-Learning: utilise max Q(s',a')
• SARSA: utilise Q(s',a') réelle

Avantages:
✓ Plus prudent (évite les risques)
✓ Stable
✓ Bon dans environnements dangereux

Idéal pour: Environnements avec pénalités""",

    'agent_dqn': """Deep Q-Network (OFF-POLICY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: OFF-POLICY avec réseau de neurones

Principe:
Utilise un réseau de neurones pour approximer
la fonction Q au lieu d'une table.

Innovations:
• Experience Replay: mémorise et rejoue
  les transitions aléatoirement
• Target Network: réseau stable pour
  les valeurs cibles

Avantages:
✓ Fonctionne sur grands espaces d'états
✓ Généralisation à états non vus
✓ Très puissant

Inconvénients:
✗ Plus lent à entraîner
✗ Plus de données nécessaires
✗ Hyperparamètres sensibles

Idéal pour: Grands problèmes complexes""",

    'alpha': """Taux d'Apprentissage (Alpha α)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 0.0 à 1.0
Recommandé: 0.1

Contrôle à quel point les nouvelles
informations remplacent les anciennes.

• Alpha ÉLEVÉ (0.5-1.0):
  ✓ Apprentissage rapide
  ✗ Instable, oublie vite
  ✗ Peut osciller

• Alpha BAS (0.01-0.1):
  ✓ Apprentissage stable
  ✗ Lent à converger
  ✗ Nécessite plus d'épisodes

• Alpha MOYEN (0.1-0.3):
  ✓ Bon équilibre
  ✓ Recommandé pour débuter

Impact:
→ Trop haut: instabilité
→ Trop bas: lenteur""",

    'gamma': """Facteur d'Actualisation (Gamma γ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 0.0 à 1.0
Recommandé: 0.95-0.99

Importance des récompenses futures
dans les décisions.

• Gamma ÉLEVÉ (0.95-0.99):
  ✓ Planification long terme
  ✓ Trouve chemins optimaux
  ✗ Apprentissage plus lent

• Gamma BAS (0.7-0.9):
  ✓ Focus court terme
  ✓ Apprentissage rapide
  ✗ Solutions sous-optimales

Pour Taxi:
• 0.99: trouve le chemin optimal
• 0.9: prend des raccourcis
• 0.7: très court terme

Impact:
→ Plus élevé = meilleur planification""",

    'epsilon': """Taux d'Exploration (Epsilon ε)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 0.0 à 1.0
Recommandé début: 1.0
Recommandé fin: 0.01

Probabilité de choisir une action
aléatoire (exploration vs exploitation).

• Epsilon = 1.0 (100%):
  → Actions 100% aléatoires
  → Maximum d'exploration

• Epsilon = 0.5 (50%):
  → 50% aléatoire, 50% optimal
  → Équilibre exploration/exploitation

• Epsilon = 0.01 (1%):
  → 99% optimal, 1% aléatoire
  → Maximum d'exploitation

Stratégie typique:
1. Début: epsilon = 1.0 (explorer)
2. Progressivement: ↓ avec decay
3. Fin: epsilon = 0.01 (exploiter)

Impact:
→ Trop haut: n'apprend pas
→ Trop bas: reste bloqué localement""",

    'epsilon_decay': """Décroissance Epsilon (Decay)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 0.9 à 1.0
Recommandé: 0.995

Facteur de réduction d'epsilon
après chaque épisode.

Formule: ε ← ε × decay

• Decay ÉLEVÉ (0.999):
  → Décroissance très lente
  → Explore longtemps
  → Nécessite beaucoup d'épisodes

• Decay MOYEN (0.995):
  → Bon équilibre
  → ~1000 épisodes pour atteindre min

• Decay BAS (0.99):
  → Décroissance rapide
  → ~500 épisodes pour atteindre min
  → Risque d'exploitation trop tôt

Exemple avec decay=0.995:
• Épisode 0: ε = 1.0
• Épisode 100: ε ≈ 0.6
• Épisode 500: ε ≈ 0.08
• Épisode 1000: ε = 0.01 (min)

Impact:
→ Contrôle la vitesse d'apprentissage""",

    'epsilon_min': """Epsilon Minimum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 0.0 à 0.2
Recommandé: 0.01

Valeur minimale d'epsilon.
Garde toujours un peu d'exploration.

• Epsilon_min = 0.0:
  → 100% exploitation
  → Aucune exploration
  → Risque de rester bloqué

• Epsilon_min = 0.01 (1%):
  → 99% exploitation, 1% exploration
  → Recommandé
  → Continue à découvrir

• Epsilon_min = 0.1 (10%):
  → 90% exploitation, 10% exploration
  → Beaucoup d'exploration résiduelle
  → Performance finale moins bonne

Pourquoi garder exploration?
→ Évite d'être piégé localement
→ S'adapte aux changements

Impact:
→ 0.0: risqué
→ 0.01-0.05: recommandé""",

    'batch_size': """Taille du Batch (DQN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 16 à 256
Recommandé: 32-64

Nombre de transitions utilisées
par mise à jour du réseau.

• Batch PETIT (16-32):
  ✓ Mises à jour fréquentes
  ✓ Apprentissage plus rapide
  ✗ Plus bruité, instable

• Batch MOYEN (32-64):
  ✓ Bon équilibre
  ✓ Recommandé

• Batch GRAND (128-256):
  ✓ Apprentissage stable
  ✗ Plus lent
  ✗ Nécessite plus de mémoire

Impact sur performance:
→ Plus grand = plus stable
→ Plus petit = plus rapide

Pour Taxi:
• Débutant: 32
• Optimal: 64-128""",

    'buffer_size': """Taille du Buffer (DQN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 1000 à 100000
Recommandé: 10000

Nombre de transitions mémorisées
dans le replay buffer.

• Buffer PETIT (1000-5000):
  ✓ Moins de mémoire
  ✗ Moins de diversité
  ✗ Oublie vite

• Buffer MOYEN (10000):
  ✓ Bon équilibre
  ✓ Recommandé pour Taxi

• Buffer GRAND (50000+):
  ✓ Maximum de diversité
  ✗ Beaucoup de mémoire
  ✗ Peut garder vieilles données

Principe:
Le buffer stocke les expériences passées
et les rejoue aléatoirement pour apprendre.

Impact:
→ Plus grand = meilleure diversité
→ Trop grand = données obsolètes

Pour Taxi:
• Minimum: 5000
• Optimal: 10000-20000""",

    'target_update': """Fréquence Update Target (DQN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plage: 50 à 1000
Recommandé: 100-200

Nombre de steps entre chaque
synchronisation du target network.

Principe:
DQN utilise 2 réseaux:
• Q-network: mis à jour constamment
• Target network: copie stable

• Update FRÉQUENT (50-100):
  ✓ Target plus à jour
  ✗ Moins stable

• Update MOYEN (100-200):
  ✓ Bon équilibre
  ✓ Recommandé

• Update RARE (500-1000):
  ✓ Très stable
  ✗ Target peut être obsolète
  ✗ Apprentissage plus lent

Pourquoi 2 réseaux?
→ Stabilise l'apprentissage
→ Évite les oscillations

Impact:
→ Plus fréquent = moins stable
→ Moins fréquent = plus lent""",

    'reward_system': """Système de Récompenses Taxi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Récompenses par défaut:
• Déposer passager correct: +20
• Prendre passager correct: 0
• Chaque step: -1
• Action illégale: -10
  (prendre/déposer au mauvais endroit)

Exemple d'épisode:
1. Start → vers passager: -10 steps = -10
2. Prendre passager: 0
3. Vers destination: -8 steps = -8
4. Déposer passager: +20
Total: +20 -10 -8 = +2

Épisode optimal: ~13 steps
Récompense optimale: ~+7 à +10

Pourquoi -1 par step?
→ Encourage efficacité
→ Trouve le chemin le plus court

Impact du système:
• Bon agent: +5 à +15 récompense
• Agent moyen: -50 à +5
• Agent aléatoire: -350 à -400""",

    'episodes': """Nombre d'Épisodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Un épisode = une partie complète
du jeu (du début au succès/échec).

Pour l'entraînement:
• Minimum: 1000 épisodes
• Recommandé: 5000 épisodes
• Optimal: 10000+ épisodes

Pour le test:
• Minimum: 100 épisodes
• Recommandé: 500 épisodes

Progression typique:
• 0-500: exploration, performance faible
• 500-2000: apprentissage, amélioration
• 2000-5000: convergence
• 5000+: fine-tuning

Plus d'épisodes:
✓ Meilleur apprentissage
✓ Plus stable
✗ Plus long

Impact:
→ Peu d'épisodes: sous-apprentissage
→ Beaucoup d'épisodes: sur-apprentissage?""",

    'mode_train': """Mode Entraînement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L'agent apprend en jouant.

Processus:
1. Agent explore (epsilon-greedy)
2. Reçoit récompenses
3. Met à jour sa politique
4. Améliore progressivement

Caractéristiques:
• Exploration active (epsilon élevé)
• Mises à jour des Q-values
• Performance augmente avec le temps

Visualisation:
• Peut être lente (calculs)
• Affiche l'apprentissage en temps réel

Recommandations:
• Commencer avec beaucoup d'épisodes
• Observer la progression
• Ajuster hyperparamètres si stagnation""",

    'mode_test': """Mode Test/Évaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L'agent utilise ce qu'il a appris
sans apprentissage supplémentaire.

Processus:
1. Agent exploite (epsilon = 0)
2. Utilise politique apprise
3. Aucune mise à jour
4. Évalue la performance

Caractéristiques:
• Pas d'exploration (actions optimales)
• Pas de mise à jour
• Performance stable

Utilité:
• Évaluer la performance réelle
• Comparer différents agents
• Valider l'apprentissage

Recommandations:
• Tester après entraînement
• Utiliser ~100-500 épisodes
• Comparer métriques""",

    'on_policy': """Algorithme ON-POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apprend la politique qu'il SUIT.

Principe:
L'agent met à jour ses valeurs en
utilisant les actions qu'il prend
réellement avec sa politique actuelle
(incluant l'exploration).

Exemple: SARSA
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
                              ↑
                    Action réelle prise

Caractéristiques:
• Apprend sa propre politique
• Plus prudent
• Tient compte de l'exploration

Avantages:
✓ Stable
✓ Sûr dans environnements risqués
✓ Performance garantie

Inconvénients:
✗ Peut être sous-optimal
✗ Plus lent à converger

Agent ON-POLICY: SARSA""",

    'off_policy': """Algorithme OFF-POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apprend une politique DIFFÉRENTE
de celle qu'il suit.

Principe:
L'agent peut explorer aléatoirement
tout en apprenant la politique optimale
(greedy).

Exemple: Q-Learning
Q(s,a) ← Q(s,a) + α[r + γ MAX Q(s',a') - Q(s,a)]
                              ↑
                    Meilleure action possible

Caractéristiques:
• Sépare exploration et apprentissage
• Apprend l'optimal même en explorant
• Plus agressif

Avantages:
✓ Converge vers l'optimal
✓ Plus rapide
✓ Réutilise expériences passées

Inconvénients:
✗ Peut être instable
✗ Risqué dans env. dangereux

Agents OFF-POLICY: Q-Learning, DQN"""
}


def add_tooltip(widget, key):
    """
    Ajoute un tooltip à un widget.

    Args:
        widget: Widget Tkinter
        key: Clé du tooltip dans TOOLTIPS
    """
    if key in TOOLTIPS:
        ToolTip(widget, TOOLTIPS[key])
