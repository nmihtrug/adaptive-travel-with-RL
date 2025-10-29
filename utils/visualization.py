"""
Visualization utilities for travel recommendation system.

Provides comprehensive plotting functions for:
- Learning curves
- User-specific performance
- Arm distributions
- Algorithm comparisons
- Regret analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from IPython.display import clear_output

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_learning_curves(
    avg_rewards: List[float],
    title: str = "Learning Curve",
    show: bool = True
):
    """
    Plot overall learning curve showing average reward over time.
    
    Args:
        avg_rewards (List[float]): Average rewards per round
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, label="Average Reward", linewidth=2)
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_user_rewards(
    user_rewards: Dict[int, List[float]],
    title: str = "Average Reward per User",
    show: bool = True
):
    """
    Plot cumulative average reward for each user.
    
    Args:
        user_rewards (Dict): Dictionary mapping user_id to list of rewards
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 5))
    
    for uid, rewards in user_rewards.items():
        if len(rewards) > 0:
            cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.plot(cumulative_avg, label=f"User {uid}", linewidth=2)
    
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Cumulative Average Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_arm_distribution_linucb(
    agent,
    places: List[Dict],
    user_id: int,
    user_profiles: List[Dict],
    place_types: List[str],
    show: bool = True
):
    """
    Visualize LinUCB scores for each arm given a specific user.
    
    Args:
        agent: LinUCBAgent instance
        places (List[Dict]): List of places
        user_id (int): User ID to analyze
        user_profiles (List[Dict]): List of user profiles
        place_types (List[str]): List of place types
        show (bool): Whether to display the plot
    """
    from data.synthetic_data import context_to_vector
    
    user = user_profiles[user_id]
    x = context_to_vector(user, place_types)

    scores = []
    for arm in range(agent.n_arms):
        A_inv = np.linalg.inv(agent.A[arm])
        theta = A_inv @ agent.b[arm]
        score = theta @ x + agent.alpha * np.sqrt(x @ A_inv @ x)
        scores.append(score)

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(places))
    bars = plt.bar(range(len(places)), scores, color=colors)
    
    plt.xticks(range(len(places)), [p['name'] for p in places], rotation=45, ha='right')
    plt.ylabel("LinUCB Score (Reward + Confidence)", fontsize=12)
    plt.xlabel("Places", fontsize=12)
    plt.title(f"LinUCB Score per Arm for User {user_id}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if show:
        plt.show()


def plot_arm_distribution_thompson(
    agent,
    places: List[Dict],
    user_id: int,
    user_profiles: List[Dict],
    place_types: List[str],
    n_samples: int = 1000,
    show: bool = True
):
    """
    Visualize posterior reward distributions for Thompson Sampling.
    
    Args:
        agent: ContextualThompsonSampling instance
        places (List[Dict]): List of places
        user_id (int): User ID to analyze
        user_profiles (List[Dict]): List of user profiles
        place_types (List[str]): List of place types
        n_samples (int): Number of samples for KDE
        show (bool): Whether to display the plot
    """
    from data.synthetic_data import context_to_vector
    
    user = user_profiles[user_id]
    x = context_to_vector(user, place_types)
    
    plt.figure(figsize=(12, 6))
    
    for arm in range(agent.n_arms):
        A_inv = np.linalg.inv(agent.A[arm])
        mu = A_inv @ agent.b[arm]
        cov = agent.alpha**2 * A_inv
        theta_samples = np.random.multivariate_normal(mu, cov, size=n_samples)
        reward_samples = theta_samples @ x
        
        sns.kdeplot(
            reward_samples,
            label=f"{places[arm]['name']} ({places[arm]['type']})",
            linewidth=2
        )
    
    plt.xlabel("Predicted Reward", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Posterior Reward Distribution per Arm for User {user_id}", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_comparison(
    results: Dict[str, Dict],
    metric: str = 'avg_reward',
    title: str = None,
    show: bool = True
):
    """
    Compare multiple agents on a specific metric.
    
    Args:
        results (Dict): Results from compare_agents function
        metric (str): Metric to compare ('avg_reward', 'total_reward')
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    if title is None:
        title = f"Agent Comparison: {metric.replace('_', ' ').title()}"
    
    agent_names = list(results.keys())
    values = [results[name][metric] for name in agent_names]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(agent_names))
    bars = plt.bar(agent_names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.xlabel("Agent", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if show:
        plt.show()


def plot_regret(
    arm_history: List[int],
    places: List[Dict],
    user_profiles: List[Dict],
    title: str = "Cumulative Regret",
    show: bool = True
):
    """
    Plot cumulative regret over time.
    
    Regret = optimal reward - actual reward
    
    Args:
        arm_history (List[int]): History of selected arms
        places (List[Dict]): List of places
        user_profiles (List[Dict]): List of user profiles
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    # Compute optimal expected reward (simplified)
    optimal_reward = 1.0  # Assume perfect matching gives reward 1
    
    cumulative_regret = []
    regret_sum = 0
    
    for arm in arm_history:
        # Simplified: assume regret based on inverse of selection count
        regret = optimal_reward * 0.1  # Placeholder
        regret_sum += regret
        cumulative_regret.append(regret_sum)
    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_regret, linewidth=2, color='crimson')
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_multiple_learning_curves(
    curves_dict: Dict[str, List[float]],
    title: str = "Learning Curves Comparison",
    show: bool = True
):
    """
    Plot multiple learning curves on the same plot for comparison.
    
    Args:
        curves_dict (Dict): Dictionary mapping agent_name to avg_rewards list
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(curves_dict))
    
    for (name, rewards), color in zip(curves_dict.items(), colors):
        plt.plot(rewards, label=name, linewidth=2, color=color)
    
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_arm_selection_frequency(
    arm_history: List[int],
    places: List[Dict],
    title: str = "Arm Selection Frequency",
    show: bool = True
):
    """
    Plot how often each arm was selected during training.
    
    Args:
        arm_history (List[int]): History of selected arms
        places (List[Dict]): List of places
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    from collections import Counter
    
    arm_counts = Counter(arm_history)
    
    plt.figure(figsize=(12, 6))
    arms = range(len(places))
    counts = [arm_counts.get(i, 0) for i in arms]
    
    colors = sns.color_palette("coolwarm", len(places))
    bars = plt.bar(arms, counts, color=colors, edgecolor='black', linewidth=1)
    
    plt.xticks(arms, [p['name'] for p in places], rotation=45, ha='right')
    plt.ylabel("Selection Count", fontsize=12)
    plt.xlabel("Places", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if show:
        plt.show()
