# Taxi Driver — Reinforcement Learning

Implémentation et comparaison de trois algorithmes de reinforcement learning sur l'environnement **Taxi-v3** (Gymnasium), avec une interface graphique interactive.

---

## Algorithmes implémentés

| Algorithme | Type | Fichier | Particularité |
|---|---|---|---|
| **Brute-Force** | Baseline | `bruteforce.py` | Actions aléatoires, aucun apprentissage |
| **Q-Learning** | Off-policy, tabulaire | `qlearning.py` | Q-table 500×6, mise à jour Bellman à chaque pas |
| **Monte Carlo** | On-policy, tabulaire | `montecarlo.py` | Attend la fin d'un épisode, retour exact G |
| **Deep Q-Network** | Off-policy, réseau de neurones | `dqn.py` | MLP NumPy, experience replay, target network |

---

## Prérequis

- Python **3.9 – 3.12**
- pip

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd T-AIA-902-Taxi-Driver
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient :

```
gymnasium==1.0.0
numpy==1.26.4
matplotlib==3.9.2
tqdm==4.66.5
```

> **Note** : le DQN est implémenté en NumPy pur — aucune dépendance PyTorch/TensorFlow requise.

---

## Lancer l'interface graphique

```bash
python src/gui.py
```

La GUI s'ouvre directement. Toutes les fonctionnalités sont accessibles depuis l'interface.

---

## Utilisation de la GUI

### Panneau gauche — Configuration

**Algorithme** : choisir entre Q-Learning, Monte Carlo ou Deep Q-Network.

**Mode** :
- `User` — vous choisissez tous les hyperparamètres manuellement
- `Time-limited` — entraîne l'agent pendant un budget temps donné (secondes)
- `Benchmark` — grid search automatique sur alpha et gamma (Q-Learning uniquement)

**Hyperparamètres communs** :
| Paramètre | Description | Par défaut |
|---|---|---|
| Alpha (lr) | Taux d'apprentissage *(Q-Learning uniquement)* | 0.1 |
| Gamma | Facteur de discount (importance du futur) | 0.99 |
| Epsilon (init) | Taux d'exploration initial | 1.0 |
| Epsilon min | Plancher d'exploration | 0.01 |
| Epsilon decay | Décroissance de l'exploration par épisode | 0.995 |

**Paramètres DQN** *(visible uniquement avec l'algo DQN)* :
| Paramètre | Description | Par défaut |
|---|---|---|
| Learning rate | Taux d'apprentissage du réseau | 0.001 |
| Batch size | Taille du mini-batch d'experience replay | 32 |
| Memory size | Capacité du replay buffer | 10 000 |
| Target update | Fréquence de mise à jour du target network (en steps) | 100 |

**Cliquer sur `▶ Lancer l'entraînement`** pour démarrer.

### Panneau central — Simulation

Après l'entraînement, cliquer sur un des boutons de visualisation pour animer un épisode complet :
- `👁 Regarder épisode (Q-Learning)` — agent Q-Learning entraîné
- `🎲 Regarder épisode (Monte Carlo)` — agent Monte Carlo entraîné
- `🧠 Regarder épisode (Deep Q-Network)` — agent DQN entraîné
- `🎲 Regarder épisode (Brute-Force)` — agent aléatoire (disponible immédiatement)

Le slider **Vitesse** contrôle la cadence de l'animation (gauche = rapide, droite = lent).

**Légende de la grille** :
- Cercles colorés `R G Y B` : stations (pickup / dropoff)
- Contour cyan pointillé : destination actuelle
- `P` blanc : passager (s'il n'est pas dans le taxi)
- Rectangle jaune : taxi (orange = passager embarqué)
- Barres rouges : murs infranchissables

### Panneau droit — Console & Graphiques

La **console** affiche en temps réel les métriques à chaque étape (brute-force, entraînement, test).

Les **courbes** montrent :
1. Reward lissé par épisode
2. Nombre de steps lissé par épisode
3. Évolution d'epsilon (exploration)

---

## Lancer depuis le terminal (sans GUI)

```bash
# Mode utilisateur
python src/main.py user --alpha 0.1 --gamma 0.99 --train 5000 --test 100

# Mode temps limité (30 secondes)
python src/main.py time --time 30 --test 100

# Mode benchmark complet
python src/main.py benchmark --train 5000 --test 100
```

---

## Structure du projet

```
T-AIA-902-Taxi-Driver/
├── src/
│   ├── environment.py      # Wrapper Gymnasium Taxi-v3
│   ├── bruteforce.py       # Agent aléatoire (baseline)
│   ├── qlearning.py        # Agent Q-Learning tabulaire
│   ├── montecarlo.py       # Agent Monte Carlo first-visit
│   ├── dqn.py              # Agent Deep Q-Network (NumPy)
│   ├── benchmark.py        # Grid search et comparaisons
│   ├── gui.py              # Interface graphique (Tkinter)
│   └── main.py             # Point d'entrée CLI
├── requirements.txt
└── README.md
```

---

## Environnement Taxi-v3

**Grille 5×5** avec 4 stations (R, G, Y, B).

| Élément | Valeur |
|---|---|
| États | 500 (25 positions × 5 emplacements passager × 4 destinations) |
| Actions | 6 (Sud, Nord, Est, Ouest, Pickup, Dropoff) |
| Reward step | −1 |
| Reward illégal | −10 |
| Reward succès | +20 |

---

## Comparaison des algorithmes

| | Brute-Force | Q-Learning | Monte Carlo | DQN |
|---|---|---|---|---|
| **Apprentissage** | Aucun | Mis à jour à chaque step | Mis à jour en fin d'épisode | Réseau de neurones |
| **Stockage** | — | Q-table 500×6 | Q-table + compteurs | Poids réseau |
| **Alpha** | — | Oui | Non (moyenne exacte) | Learning rate réseau |
| **Biais** | — | Oui (estimation bootstrap) | Non (retour exact) | Oui |
| **Vitesse de convergence** | — | Rapide | Plus lente | Moyenne |

---

## Métriques reportées

- **Reward moyen** et écart-type
- **Steps moyens** et écart-type
- **Taux de succès** (% d'épisodes avec reward > 0)
- **Reward par step**
- **Actions illégales** totales
- **Épisode de convergence** (premier épisode où 25% des 50 derniers sont réussis)
