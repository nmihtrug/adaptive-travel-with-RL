"""
Utilities package for visualization, interactive features, and persistence.
"""

from .visualization import (
    plot_learning_curves,
    plot_user_rewards,
    plot_arm_distribution_linucb,
    plot_arm_distribution_thompson,
    plot_comparison,
    plot_regret
)

from .interactive import (
    interactive_recommendation_egreedy,
    interactive_recommendation_linucb,
    interactive_recommendation_thompson
)

from .persistence import (
    save_agent,
    load_agent,
    save_training_history,
    load_training_history,
    save_metrics,
    load_metrics,
    save_dataset,
    load_dataset,
    save_complete_experiment,
    load_experiment
)

__all__ = [
    'plot_learning_curves',
    'plot_user_rewards',
    'plot_arm_distribution_linucb',
    'plot_arm_distribution_thompson',
    'plot_comparison',
    'plot_regret',
    'interactive_recommendation_egreedy',
    'interactive_recommendation_linucb',
    'interactive_recommendation_thompson',
    'save_agent',
    'load_agent',
    'save_training_history',
    'load_training_history',
    'save_metrics',
    'load_metrics',
    'save_dataset',
    'load_dataset',
    'save_complete_experiment',
    'load_experiment'
]
