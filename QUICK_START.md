# 🚀 Démarrage Rapide - Taxi Driver RL

## Installation (1 minute)

```bash
# 1. Installe les dépendances
pip install gymnasium numpy torch matplotlib pandas

# 2. Lance l'application
python main.py
```

## Premier Test (5 minutes)

### Étape 1: Comprends l'UI

```
┌─────────────────────────────────────────────────────────┐
│  🚕 TAXI DRIVER - RL Interactive                        │
├──────────────────┬──────────────────────────────────────┤
│                  │                                      │
│  🎮 Grille 5x5   │  🎛️ Hyperparamètres               │
│                  │                                      │
│  ⚙️ Contrôles    │  💡 Recommandations                │
│                  │                                      │
└──────────────────┴──────────────────────────────────────┘
```

### Étape 2: Lance ton Premier Entraînement

1. **Agent**: Laisse **Q-Learning** (recommandé)
2. **Mode**: Laisse **Entraînement**
3. **Épisodes**: Mets **1000**
4. **Clique**: ▶ DÉMARRER

**Observe**:
- La grille s'anime
- Les métriques se mettent à jour
- La barre de progression avance

### Étape 3: Lis les Recommandations

Une fois terminé, regarde la section **💡 Recommandations**:

```
===========================================================
ACTIONS A FAIRE MAINTENANT:
===========================================================

1. AJUSTEZ les hyperparametres suivants:

   -> EPSILON: Mettez a 0.5
      (actuellement: 0.010)

2. Relancez l'entrainement avec 2000+ episodes
```

### Étape 4: Applique les Recommandations

1. **Trouve le slider EPSILON** dans la section Hyperparamètres
2. **Clique sur ❓** à côté pour comprendre
3. **Déplace le slider à 0.5**
4. **Clique ▶ DÉMARRER** à nouveau

### Étape 5: Compare les Résultats

Regarde **📊 Métriques Performance**:

**Avant**:
```
Récompense moyenne: -180.50
Steps moyens: 320.20
Taux de succès: 15.2%
```

**Après**:
```
Récompense moyenne: +2.30
Steps moyens: 85.40
Taux de succès: 62.8%
```

**C'est une ÉNORME amélioration! 🎉**

## Info-Bulles ❓ - Ton Meilleur Ami

**Survole les ❓** pour tout comprendre:

```
Alpha (α) ❓ (Learning Rate)
  ↑
  Clique ici pour voir:
  - Ce que fait Alpha
  - Quand l'augmenter
  - Impact sur l'apprentissage
```

### Info-Bulles Disponibles:

**Agents**:
- Random ❓
- Q-Learning ❓ → OFF-POLICY
- SARSA ❓ → ON-POLICY
- DQN ❓ → OFF-POLICY profond

**Hyperparamètres**:
- Alpha ❓
- Gamma ❓
- Epsilon ❓
- Epsilon Decay ❓
- Epsilon Min ❓
- Batch Size ❓ (DQN)
- Buffer Size ❓ (DQN)
- Target Update ❓ (DQN)

**Concepts**:
- Épisodes ❓
- Mode Entraînement ❓
- Mode Test ❓
- Système de Récompenses ❓

## Objectif: Agent Optimal

**But**: Atteindre ces métriques:
```
✓ Récompense moyenne: > +5
✓ Steps moyens: < 30
✓ Taux de succès: > 90%
```

## Stratégie Recommandée

### Phase 1: Exploration (0-2000 épisodes)
```
Agent: Q-Learning
Epsilon: 1.0 → 0.1 (exploration)
Alpha: 0.1
Gamma: 0.99
Episodes: 2000
```

### Phase 2: Exploitation (2000-5000 épisodes)
```
Epsilon: 0.1 → 0.01 (exploitation)
Episodes: 3000+
```

### Phase 3: Test
```
Mode: Test
Episodes: 500
```

## Problèmes Courants

### ❌ L'agent n'apprend pas (récompense ~-300)

**Solution**:
- Augmente **EPSILON à 0.5** (plus d'exploration)
- Augmente **ALPHA à 0.2** (apprentissage plus rapide)
- Lance **2000+ épisodes**

### ❌ Performance instable (oscille beaucoup)

**Solution**:
- Réduis **ALPHA à 0.05** (plus stable)
- Augmente **EPSILON DECAY à 0.998** (réduit exploration plus lentement)

### ❌ Agent bloqué à ~60% succès

**Solution**:
- Augmente **GAMMA à 0.99** (planification long terme)
- Réduis **EPSILON à 0.01** (plus d'exploitation)
- Lance **5000+ épisodes**

## Comparer ON-POLICY vs OFF-POLICY

### Test 1: Q-Learning (OFF-POLICY)
```
Agent: Q-Learning
Episodes: 2000
Note: Résultats
```

### Test 2: SARSA (ON-POLICY)
```
Agent: SARSA
Mêmes hyperparamètres
Episodes: 2000
Note: Résultats
```

**Compare**:
- Q-Learning devrait converger plus vite
- SARSA devrait être plus stable

## Raccourcis Clavier

- **Molette**: Scroll vertical
- **Échap**: Ferme l'application (bientôt)

## FAQ Rapide

**Q: Combien d'épisodes minimum?**
R: 1000 pour voir l'apprentissage, 5000 pour agent optimal

**Q: Quel agent choisir?**
R: Q-Learning pour débuter, DQN pour problèmes complexes

**Q: C'est quoi OFF-POLICY?**
R: Clique sur ❓ à côté de "Q-Learning" dans l'UI!

**Q: L'UI est lente?**
R: Réduis la vitesse de visualisation (slider Vitesse)

**Q: Comment sauvegarder un agent?**
R: Bientôt disponible (v2.0)

## Prochaines Étapes

1. ✅ Comprends tous les hyperparamètres (survole les ❓)
2. ✅ Atteins 90% de succès avec Q-Learning
3. ✅ Compare Q-Learning vs SARSA
4. ✅ Essaye DQN (plus puissant mais plus lent)
5. ✅ Lis le README.md complet pour approfondir

## Support

**Problème avec l'UI?**
- Vérifie que tu as Python 3.8+
- Réinstalle: `pip install --upgrade gymnasium numpy torch`
- Lance le test: `python test_setup.py`

**Questions RL?**
- Utilise les info-bulles ❓
- Lis README.md
- Lis FEATURES.md

---

**Tu es prêt! Lance `python main.py` et amuse-toi! 🎮🚀**
