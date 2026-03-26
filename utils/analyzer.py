import numpy as np


class HyperparameterAnalyzer:
    """
    Analyseur intelligent qui fournit des recommandations d'hyperparamètres
    basées sur les performances de l'agent.

    Cette classe analyse les résultats d'entraînement et suggère des ajustements
    d'hyperparamètres pour améliorer les performances.
    """

    def __init__(self):
        self.history = []

    def analyze_performance(self, agent, training_stats, current_hyperparams):
        """
        Analyse les performances et génère des recommandations.

        Args:
            agent: L'agent utilisé
            training_stats: Statistiques d'entraînement (rewards, steps, success_rate)
            current_hyperparams: Hyperparamètres actuels

        Returns:
            dict: Recommandations avec explications
        """
        recommendations = {
            'status': '',
            'suggestions': [],
            'warnings': []
        }

        if not training_stats['rewards'] or len(training_stats['rewards']) < 10:
            recommendations['status'] = 'Pas assez de données pour analyser'
            recommendations['suggestions'].append(
                "Lancez plus d'épisodes (au moins 100) pour obtenir des recommandations pertinentes."
            )
            return recommendations

        # Calcule les métriques
        recent_rewards = training_stats['rewards'][-100:]  # 100 derniers épisodes
        recent_steps = training_stats['steps'][-100:]
        recent_success = training_stats['success_rate'][-100:]

        avg_reward = np.mean(recent_rewards)
        avg_steps = np.mean(recent_steps)
        success_rate = np.mean(recent_success)

        # Calcule la tendance (amélioration ou stagnation)
        if len(training_stats['rewards']) >= 200:
            early_rewards = training_stats['rewards'][-200:-100]
            improvement = np.mean(recent_rewards) - np.mean(early_rewards)
        else:
            improvement = 0

        # Analyse générale
        if success_rate > 0.9 and avg_steps < 30:
            recommendations['status'] = '✓ Excellent! Agent très performant'
        elif success_rate > 0.7 and avg_steps < 50:
            recommendations['status'] = '✓ Bien! Agent performant'
        elif success_rate > 0.5:
            recommendations['status'] = '⚠ Moyen - Amélioration possible'
        else:
            recommendations['status'] = '✗ Faible - Ajustements nécessaires'

        # Recommandations spécifiques par type d'agent
        agent_type = agent.__class__.__name__

        if agent_type == "RandomAgent":
            recommendations['suggestions'].append(
                "Agent aléatoire: pas d'apprentissage possible. Utilisez un autre agent (Q-Learning, SARSA, DQN)."
            )
            return recommendations

        # Analyse de l'exploration (epsilon)
        epsilon = current_hyperparams.get('epsilon', 0)

        if success_rate < 0.3:
            recommendations['warnings'].append(
                "⚠ Taux de succès très faible. L'agent n'apprend pas efficacement."
            )

            # Problème d'exploration vs exploitation
            if epsilon < 0.1:
                recommendations['suggestions'].append(
                    f"• EPSILON trop bas ({epsilon:.3f}): Augmentez epsilon à 0.3-0.5 pour plus d'exploration. "
                    f"L'agent exploite trop tôt sans avoir exploré suffisamment."
                )
            elif epsilon > 0.8:
                recommendations['suggestions'].append(
                    f"• EPSILON trop élevé ({epsilon:.3f}): L'agent explore trop et n'exploite pas assez. "
                    f"Réduisez epsilon à 0.5 ou augmentez epsilon_decay."
                )

            # Problème de learning rate
            alpha = current_hyperparams.get('alpha', 0)
            if alpha < 0.05:
                recommendations['suggestions'].append(
                    f"• ALPHA (learning rate) trop bas ({alpha:.3f}): Augmentez à 0.1-0.3. "
                    f"L'agent apprend trop lentement."
                )
            elif alpha > 0.5:
                recommendations['suggestions'].append(
                    f"• ALPHA trop élevé ({alpha:.3f}): Réduisez à 0.1-0.3. "
                    f"L'apprentissage est trop instable."
                )

        elif success_rate < 0.7:
            recommendations['warnings'].append(
                "⚠ Performance moyenne. Quelques ajustements peuvent aider."
            )

            if improvement < 0 and epsilon > 0.5:
                recommendations['suggestions'].append(
                    f"• EPSILON encore élevé ({epsilon:.3f}): Augmentez epsilon_decay pour réduire "
                    f"l'exploration plus rapidement. L'agent a besoin d'exploiter plus."
                )

            if avg_steps > 100:
                gamma = current_hyperparams.get('gamma', 0)
                if gamma < 0.9:
                    recommendations['suggestions'].append(
                        f"• GAMMA (discount factor) bas ({gamma:.3f}): Augmentez à 0.95-0.99. "
                        f"L'agent doit mieux prendre en compte les récompenses futures pour planifier."
                    )

        else:
            # Performance correcte, optimisation fine
            if epsilon > 0.1:
                recommendations['suggestions'].append(
                    f"• EPSILON encore à {epsilon:.3f}: Vous pouvez le réduire à 0.01-0.05 "
                    f"pour maximiser l'exploitation maintenant que l'agent a bien appris."
                )

            if avg_steps > 30:
                recommendations['suggestions'].append(
                    "• Nombre de steps encore élevé: Augmentez le nombre d'épisodes d'entraînement "
                    "pour raffiner la politique."
                )

        # Recommandations spécifiques DQN
        if agent_type == "DQNAgent":
            batch_size = current_hyperparams.get('batch_size', 32)
            buffer_size = current_hyperparams.get('buffer_size', 10000)

            if success_rate < 0.5:
                if batch_size < 32:
                    recommendations['suggestions'].append(
                        f"• BATCH_SIZE petit ({batch_size}): Augmentez à 64-128 pour "
                        f"un apprentissage plus stable."
                    )
                if buffer_size < 5000:
                    recommendations['suggestions'].append(
                        f"• BUFFER_SIZE petit ({buffer_size}): Augmentez à 10000-20000 "
                        f"pour plus de diversité dans l'apprentissage."
                    )

        # Analyse du gamma
        gamma = current_hyperparams.get('gamma', 0)
        if avg_steps > 100 and gamma < 0.95:
            recommendations['suggestions'].append(
                f"• GAMMA ({gamma:.3f}): Augmentez à 0.95-0.99. Un gamma élevé aide l'agent "
                f"à planifier des trajectoires longues et optimales."
            )

        # Stagnation
        if improvement < -1 and len(training_stats['rewards']) > 200:
            recommendations['warnings'].append(
                "⚠ L'agent régresse ou stagne. Possible surapprentissage ou epsilon trop bas."
            )
            recommendations['suggestions'].append(
                "• Essayez de réinitialiser epsilon à 0.3 pour re-explorer, "
                "ou relancez l'entraînement avec de meilleurs hyperparamètres."
            )

        # Recommendations générales
        if not recommendations['suggestions']:
            recommendations['suggestions'].append(
                "✓ Les hyperparamètres semblent bien équilibrés. "
                "Continuez l'entraînement pour améliorer progressivement."
            )

        # Ajoute les métriques
        recommendations['metrics'] = {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'improvement': improvement
        }

        return recommendations

    def get_recommended_hyperparameters(self, agent_type, problem_type='default'):
        """
        Retourne des hyperparamètres recommandés pour un agent donné.

        Args:
            agent_type: Type d'agent ('QLearningAgent', 'SARSAAgent', 'DQNAgent')
            problem_type: 'default', 'fast_learning', 'safe_learning', 'optimal'

        Returns:
            dict: Hyperparamètres recommandés
        """
        recommendations = {}

        if agent_type in ['QLearningAgent', 'SARSAAgent']:
            if problem_type == 'fast_learning':
                recommendations = {
                    'alpha': 0.3,
                    'gamma': 0.95,
                    'epsilon': 0.8,
                    'epsilon_decay': 0.99,
                    'epsilon_min': 0.01
                }
            elif problem_type == 'safe_learning':
                recommendations = {
                    'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon': 0.5,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.05
                }
            elif problem_type == 'optimal':
                recommendations = {
                    'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.998,
                    'epsilon_min': 0.01
                }
            else:  # default
                recommendations = {
                    'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.01
                }

        elif agent_type == 'DQNAgent':
            if problem_type == 'fast_learning':
                recommendations = {
                    'alpha': 0.001,
                    'gamma': 0.95,
                    'epsilon': 0.8,
                    'epsilon_decay': 0.99,
                    'epsilon_min': 0.01,
                    'batch_size': 64,
                    'buffer_size': 5000,
                    'target_update': 100
                }
            elif problem_type == 'optimal':
                recommendations = {
                    'alpha': 0.0005,
                    'gamma': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.998,
                    'epsilon_min': 0.01,
                    'batch_size': 128,
                    'buffer_size': 20000,
                    'target_update': 200
                }
            else:  # default
                recommendations = {
                    'alpha': 0.001,
                    'gamma': 0.99,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.01,
                    'batch_size': 32,
                    'buffer_size': 10000,
                    'target_update': 100
                }

        return recommendations
