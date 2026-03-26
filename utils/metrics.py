import numpy as np


class MetricsCalculator:
    """
    Calcule diverses métriques de performance pour les agents RL.
    """

    @staticmethod
    def calculate_metrics(training_stats, window=100):
        """
        Calcule les métriques de performance.

        Args:
            training_stats: Dictionnaire avec 'rewards', 'steps', 'success_rate'
            window: Taille de la fenêtre pour les moyennes mobiles

        Returns:
            dict: Métriques calculées
        """
        if not training_stats['rewards']:
            return {
                'avg_reward': 0,
                'avg_steps': 0,
                'success_rate': 0,
                'total_episodes': 0,
                'best_reward': 0,
                'worst_reward': 0,
                'recent_avg_reward': 0,
                'recent_avg_steps': 0,
                'recent_success_rate': 0
            }

        rewards = np.array(training_stats['rewards'])
        steps = np.array(training_stats['steps'])
        success = np.array(training_stats['success_rate'])

        # Métriques globales
        metrics = {
            'avg_reward': float(np.mean(rewards)),
            'avg_steps': float(np.mean(steps)),
            'success_rate': float(np.mean(success)),
            'total_episodes': len(rewards),
            'best_reward': float(np.max(rewards)),
            'worst_reward': float(np.min(rewards)),
        }

        # Métriques récentes (dernière fenêtre)
        if len(rewards) >= window:
            metrics['recent_avg_reward'] = float(np.mean(rewards[-window:]))
            metrics['recent_avg_steps'] = float(np.mean(steps[-window:]))
            metrics['recent_success_rate'] = float(np.mean(success[-window:]))
        else:
            metrics['recent_avg_reward'] = metrics['avg_reward']
            metrics['recent_avg_steps'] = metrics['avg_steps']
            metrics['recent_success_rate'] = metrics['success_rate']

        return metrics

    @staticmethod
    def format_metrics_text(metrics):
        """
        Formate les métriques en texte lisible.

        Args:
            metrics: Dictionnaire de métriques

        Returns:
            str: Texte formaté
        """
        text = f"""
╔═══════════════════════════════════════╗
║       MÉTRIQUES DE PERFORMANCE        ║
╠═══════════════════════════════════════╣
║ Épisodes totaux: {metrics['total_episodes']:>18} ║
║                                       ║
║ MOYENNES GLOBALES                     ║
║ • Récompense moyenne: {metrics['avg_reward']:>14.2f} ║
║ • Steps moyens: {metrics['avg_steps']:>20.2f} ║
║ • Taux de succès: {metrics['success_rate']*100:>18.1f}% ║
║                                       ║
║ PERFORMANCE RÉCENTE (100 derniers)    ║
║ • Récompense moyenne: {metrics['recent_avg_reward']:>14.2f} ║
║ • Steps moyens: {metrics['recent_avg_steps']:>20.2f} ║
║ • Taux de succès: {metrics['recent_success_rate']*100:>18.1f}% ║
║                                       ║
║ EXTRÊMES                              ║
║ • Meilleure récompense: {metrics['best_reward']:>12.2f} ║
║ • Pire récompense: {metrics['worst_reward']:>17.2f} ║
╚═══════════════════════════════════════╝
"""
        return text
