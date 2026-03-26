# 🚕 Taxi Driver - Reinforcement Learning Interactive

Application interactive complète pour entraîner et tester des algorithmes de Reinforcement Learning sur l'environnement Taxi-v3 de Gymnasium.

> **✨ NOUVELLE UI v2.0**: Design moderne responsive avec icônes ❓ pour info-bulles et recommandations intelligentes avec valeurs exactes!

## 📋 Description du Projet

Ce projet implémente une interface graphique interactive permettant de:
- ✅ Entraîner et tester **4 agents RL**: Random (baseline), Q-Learning (off-policy), SARSA (on-policy), et DQN (deep off-policy)
- ✅ Ajuster les **hyperparamètres en temps réel** avec des sliders intuitifs
- ✅ Visualiser l'apprentissage sur une **grille 5x5** animée
- ✅ Obtenir des **recommandations intelligentes** d'ajustement d'hyperparamètres selon les résultats
- ✅ Comparer les performances **on-policy vs off-policy**
- ✅ Comprendre les algorithmes grâce à des **info-bulles explicatives** détaillées

## 🎯 Objectif

Résoudre le jeu Taxi-v3 où un taxi doit:
1. Récupérer un passager à un emplacement aléatoire (R, G, Y, B)
2. Le déposer à sa destination

**Performance attendue:**
- Agent aléatoire: ~350-400 steps, récompense -350 à -400
- Agent optimisé: ~13-20 steps, récompense +5 à +15

## 🚀 Installation & Démarrage Rapide

### Prérequis
- Python 3.8+
- pip

### Installation (2 minutes)

```bash
# Installe les dépendances
pip install gymnasium numpy torch matplotlib pandas

# OU avec requirements.txt
pip install -r requirements.txt

# Lance l'application
python main.py
```

**Démarrage rapide**: Voir [QUICK_START.md](QUICK_START.md) pour un guide en 5 minutes!

## 🎮 Utilisation

### Lancer l'application

```bash
python main.py
```

**Nouvelle UI v2.0**:
- ✨ Design moderne avec thème sombre
- 📱 Responsive (s'adapte à ton écran)
- ❓ Icônes cliquables pour toutes les info-bulles
- 🎯 Recommandations avec valeurs exactes à optimiser

Voir [FEATURES.md](FEATURES.md) pour tous les détails de la nouvelle UI!

### Interface Utilisateur

L'application comprend:

#### 1. **Panneau Gauche - Visualisation**
- Grille 5x5 animée montrant:
  - **T**: Taxi (carré jaune)
  - **P**: Passager (cercle bleu)
  - **★**: Destination (étoile verte)
  - **R/G/Y/B**: Emplacements spéciaux

#### 2. **Contrôles**
- **Choix de l'agent**: Random, Q-Learning, SARSA, DQN
- **Mode**: Entraînement (apprentissage) ou Test (évaluation)
- **Épisodes**: Nombre d'épisodes à exécuter
- **Vitesse**: Vitesse de visualisation

#### 3. **Panneau Droit - Hyperparamètres et Résultats**
- **Hyperparamètres ajustables**:
  - **Alpha (α)**: Taux d'apprentissage (0.01-1.0)
  - **Gamma (γ)**: Facteur d'actualisation (0.5-1.0)
  - **Epsilon (ε)**: Taux d'exploration (0.0-1.0)
  - **Epsilon Decay**: Décroissance de l'exploration (0.9-1.0)
  - **Epsilon Min**: Exploration minimale (0.0-0.5)
  - **DQN uniquement**: Batch Size, Buffer Size, Target Update

- **Métriques de Performance**:
  - Récompense moyenne
  - Nombre de steps moyen
  - Taux de succès
  - Statistiques récentes vs globales

- **Recommandations Intelligentes**:
  - Analyse automatique des résultats
  - Suggestions d'ajustement d'hyperparamètres
  - Explications des problèmes détectés

### 📊 Workflow Recommandé

1. **Choisir un agent** (commencer avec Q-Learning)
2. **Utiliser les valeurs par défaut** (bouton "Réinitialiser aux valeurs recommandées")
3. **Lancer l'entraînement** avec 1000-5000 épisodes
4. **Observer les métriques** pendant l'entraînement
5. **Lire les recommandations** une fois terminé
6. **Ajuster les hyperparamètres** selon les suggestions
7. **Réentraîner** et comparer les résultats
8. **Tester l'agent** en mode Test pour évaluer la performance finale

## 🧠 Algorithmes Implémentés

### 1. **Random Agent (Baseline)**
- **Type**: Aucun apprentissage
- **Description**: Choisit des actions aléatoires
- **Utilité**: Référence de comparaison

### 2. **Q-Learning (OFF-POLICY)** ⭐ Recommandé pour débuter
- **Type**: OFF-POLICY
- **Principe**: Apprend la politique optimale indépendamment de celle suivie
- **Formule**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- **Avantages**: Simple, efficace, converge vers l'optimal
- **Hyperparamètres recommandés**:
  - Alpha: 0.1
  - Gamma: 0.99
  - Epsilon: 1.0 → 0.01

### 3. **SARSA (ON-POLICY)**
- **Type**: ON-POLICY
- **Principe**: Apprend la politique qu'il suit actuellement
- **Formule**: `Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]`
- **Avantages**: Plus prudent, stable
- **Différence avec Q-Learning**: Utilise l'action réellement prise, pas la meilleure

### 4. **Deep Q-Network (DQN) (OFF-POLICY)**
- **Type**: OFF-POLICY avec réseau de neurones
- **Principe**: Approxime la fonction Q avec un réseau de neurones
- **Innovations**:
  - Experience Replay: Rejoue les expériences passées
  - Target Network: Réseau stable pour les valeurs cibles
- **Avantages**: Puissant, fonctionne sur grands espaces d'états
- **Inconvénients**: Plus lent, nécessite plus de données

## 📖 Comprendre les Hyperparamètres

### **Alpha (α) - Taux d'Apprentissage**
- **Plage**: 0.01 à 1.0
- **Recommandé**: 0.1-0.3
- **Rôle**: Contrôle à quel point les nouvelles informations remplacent les anciennes
- **Impact**:
  - Trop élevé → Apprentissage instable, oublie vite
  - Trop bas → Apprentissage très lent

### **Gamma (γ) - Facteur d'Actualisation**
- **Plage**: 0.5 à 1.0
- **Recommandé**: 0.95-0.99
- **Rôle**: Importance des récompenses futures
- **Impact**:
  - Élevé (0.99) → Planifie à long terme, trouve chemins optimaux
  - Bas (0.7) → Focus court terme, solutions sous-optimales

### **Epsilon (ε) - Taux d'Exploration**
- **Plage**: 0.0 à 1.0
- **Recommandé début**: 1.0 (100% exploration)
- **Recommandé fin**: 0.01 (1% exploration)
- **Rôle**: Probabilité de choisir une action aléatoire
- **Stratégie**: Commencer élevé, réduire progressivement (epsilon decay)

### **Epsilon Decay - Décroissance de l'Exploration**
- **Plage**: 0.9 à 1.0
- **Recommandé**: 0.995
- **Rôle**: Facteur de réduction d'epsilon après chaque épisode
- **Formule**: `ε ← ε × decay`
- **Exemple**: Avec decay=0.995, epsilon passe de 1.0 à 0.01 en ~1000 épisodes

## 🔬 Système de Récompenses

L'environnement Taxi-v3 utilise:
- **+20**: Déposer le passager à la bonne destination
- **-1**: Chaque step (encourage l'efficacité)
- **-10**: Action illégale (prendre/déposer au mauvais endroit)

**Exemple d'épisode optimal:**
- 10 steps vers passager: -10
- Prendre passager: 0
- 8 steps vers destination: -8
- Déposer passager: +20
- **Total: +2 points**

## 📈 Interpréter les Résultats

### **Métriques Clés**

1. **Récompense Moyenne**
   - < -100: Très mauvais (proche du random)
   - -50 à 0: Mauvais
   - 0 à +5: Moyen
   - +5 à +15: Bon
   - > +15: Excellent (optimal)

2. **Steps Moyens**
   - > 200: Très mauvais
   - 100-200: Mauvais
   - 50-100: Moyen
   - 20-50: Bon
   - < 20: Excellent (optimal ~13 steps)

3. **Taux de Succès**
   - < 30%: Mauvais
   - 30-70%: Moyen
   - 70-90%: Bon
   - > 90%: Excellent

### **Problèmes Courants et Solutions**

#### **Problème: Agent n'apprend pas (récompense stagne à -200)**
**Causes possibles:**
- Alpha trop bas → Augmenter à 0.1-0.3
- Epsilon trop bas → Augmenter à 0.5-1.0 pour plus d'exploration
- Pas assez d'épisodes → Augmenter à 5000+

#### **Problème: Performance instable (oscille beaucoup)**
**Causes possibles:**
- Alpha trop élevé → Réduire à 0.1
- Epsilon decay trop rapide → Augmenter à 0.997-0.999

#### **Problème: Agent stagne à une solution sous-optimale**
**Causes possibles:**
- Gamma trop bas → Augmenter à 0.99
- Epsilon trop bas → Réinitialiser epsilon à 0.3 pour re-explorer

## 🎓 Comparer ON-POLICY vs OFF-POLICY

### **Expérience Suggérée**

1. **Entraîner Q-Learning** (OFF-POLICY) sur 2000 épisodes
   - Noter: récompense finale, steps, taux de succès

2. **Entraîner SARSA** (ON-POLICY) sur 2000 épisodes avec mêmes hyperparamètres
   - Noter: récompense finale, steps, taux de succès

3. **Comparer**:
   - Q-Learning devrait converger plus vite et atteindre de meilleures performances
   - SARSA devrait être plus stable mais potentiellement moins optimal

### **Différence Clé**

- **Q-Learning**: Utilise `max Q(s',a')` → Apprend l'optimal même en explorant
- **SARSA**: Utilise `Q(s',a')` réelle → Apprend ce qu'il fait vraiment

## 🏆 Challenge

**Objectif**: Entraîner un agent qui atteint:
- Récompense moyenne > +10
- Steps moyens < 20
- Taux de succès > 95%

**Astuces**:
1. Utiliser Q-Learning ou DQN
2. Entraîner sur 10000+ épisodes
3. Utiliser gamma = 0.99
4. Commencer avec epsilon = 1.0, decay = 0.998

## 📁 Structure du Projet

```
T-AIA-902-Taxi-Driver/
│
├── agents/                      # Agents RL
│   ├── __init__.py
│   ├── base.py                 # Classe de base
│   ├── random_agent.py         # Agent aléatoire
│   ├── q_learning.py           # Q-Learning (off-policy)
│   ├── sarsa.py                # SARSA (on-policy)
│   └── dqn.py                  # Deep Q-Network (off-policy)
│
├── ui/                          # Interface utilisateur
│   ├── __init__.py
│   ├── main_window.py          # Fenêtre principale
│   ├── taxi_grid.py            # Visualisation grille 5x5
│   └── tooltips.py             # Système de tooltips
│
├── utils/                       # Utilitaires
│   ├── __init__.py
│   ├── analyzer.py             # Analyseur intelligent
│   └── metrics.py              # Calculateur de métriques
│
├── main.py                      # Point d'entrée
├── requirements.txt             # Dépendances
└── README.md                    # Ce fichier
```

## 💡 Info-Bulles (Tooltips)

L'application contient des **info-bulles détaillées** sur:
- Chaque agent (fonctionnement, avantages, inconvénients)
- Chaque hyperparamètre (rôle, impact, valeurs recommandées)
- Les concepts (on-policy, off-policy, système de récompenses)

**Comment les utiliser**: Survolez n'importe quel élément avec la souris pendant 0.5s

## 🔧 Personnalisation

### Modifier les Hyperparamètres par Défaut

Éditez `utils/analyzer.py`, fonction `get_recommended_hyperparameters()`.

### Ajouter un Nouvel Agent

1. Créer un fichier dans `agents/` héritant de `BaseAgent`
2. Implémenter les méthodes abstraites
3. Ajouter dans `agents/__init__.py`
4. Ajouter dans `ui/main_window.py`

## 📚 Ressources

- [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/)
- [Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
- [SARSA](https://en.wikipedia.org/wiki/State-Action-Reward-State-Action)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

## 👨‍💻 Auteur

Projet développé dans le cadre du cours T-AIA-902 - Epitech

## 📝 Licence

Ce projet est à usage éducatif.

---

**Bon apprentissage par renforcement! 🚀**
