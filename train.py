"""
Training Script for Adaptive Travel Recommendation with RL

This script trains three reinforcement learning models:
- Epsilon-Greedy
- LinUCB (Contextual Bandit)
- Thompson Sampling (Contextual)

Usage:
    python train.py [--rounds ROUNDS] [--save]
"""

import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from models.agents import EpsilonGreedyAgent, LinUCBAgent, ContextualThompsonSampling


def load_data():
    """Load user profiles and places data."""
    print("Loading data...")
    
    # Load user profiles
    df_users = pd.read_csv('data/gen/user_profiles.csv')
    print(f"✓ Loaded {len(df_users)} user profiles")
    
    # Extract feature names
    feature_columns = [col for col in df_users.columns if col != 'user_id']
    
    # Load places from final_dataset.csv
    df_places = pd.read_csv('data/final_dataset.csv')
    print(f"✓ Loaded {len(df_places)} places")
    
    # Convert dataset to places format
    places = []
    for _, row in df_places.iterrows():
        keywords = row['Keywords'].split(', ')
        primary_type = keywords[0] if keywords else 'du lịch'
        places.append({
            "name": row['Location Name'],
            "type": primary_type,
            "keywords": keywords,
            "rating": row['Rating']
        })
    
    # Convert DataFrame to user profile dictionaries
    user_profiles = []
    for _, row in df_users.iterrows():
        user_id = int(row['user_id'])
        prefs = {col: row[col] for col in feature_columns}
        user_profiles.append({"id": user_id, "prefs": prefs})
    
    return user_profiles, places, feature_columns


def simulate_reward(user, place):
    """
    Simulate user reward based on preference match and place rating.
    
    Reward = 1 if user clicks/rates high, otherwise low
    Probability of reward depends on preference match between user and place
    """
    pref_score = user["prefs"].get(place["type"], 0)
    place_r = np.clip(place.get("rating", 3.0) / 5.0, 0, 1)
    utility = 0.7 * pref_score + 0.3 * place_r
    prob = np.clip(utility + np.random.normal(0, 0.05), 0, 1)
    
    rating = int(np.floor(prob * 5)) + 1
    rating = max(1, min(5, rating))
    
    return rating


def context_to_vector(user, feature_columns):
    """Convert user preferences to vector."""
    return np.array([user["prefs"].get(col, 0) for col in feature_columns])


def train_epsilon_greedy(user_profiles, places, n_rounds=1000, verbose=True):
    """Train Epsilon-Greedy agent."""
    if verbose:
        print("\n" + "=" * 60)
        print("Training Epsilon-Greedy Agent")
        print("=" * 60)
    
    agent = EpsilonGreedyAgent(n_arms=len(places), epsilon=0.2)
    user_rewards = {u["id"]: [] for u in user_profiles}
    avg_rewards = []
    arm_history = []
    
    for round_num in range(n_rounds):
        user = random.choice(user_profiles)
        ranked = agent.select_arm()
        
        # Simulate user interaction
        chosen_arm = random.choice(ranked)
        rating_sim = simulate_reward(user, places[chosen_arm])
        rating_norm = (rating_sim - 1) / 4
        reward = 1.0 + rating_norm
        
        # Update chosen arm
        agent.update(chosen_arm, reward)
        user_rewards[user["id"]].append(reward)
        arm_history.append(chosen_arm)
        
        # Penalize non-chosen arms
        for arm in ranked:
            if arm != chosen_arm:
                agent.update(arm, -0.3)
                user_rewards[user["id"]].append(-0.3)
                arm_history.append(arm)
        
        avg_rewards.append(np.mean([r for lst in user_rewards.values() for r in lst]))
        
        if verbose and (round_num + 1) % 500 == 0:
            print(f"Round {round_num + 1}/{n_rounds} - Avg Reward: {np.mean(avg_rewards[-100:]):.3f}")
    
    final_avg = np.mean(avg_rewards[-100:])
    if verbose:
        print(f"✓ Training complete! Final avg reward: {final_avg:.3f}")
    
    return agent, user_rewards, avg_rewards, arm_history


def train_linucb(user_profiles, places, feature_columns, n_rounds=1000, verbose=True):
    """Train LinUCB agent."""
    if verbose:
        print("\n" + "=" * 60)
        print("Training LinUCB Agent")
        print("=" * 60)
    
    agent = LinUCBAgent(n_arms=len(places), n_features=len(feature_columns), alpha=0.1)
    user_rewards = {u["id"]: [] for u in user_profiles}
    avg_rewards = []
    arm_history = []
    
    for round_num in range(n_rounds):
        user = random.choice(user_profiles)
        x = context_to_vector(user, feature_columns)
        
        # Select top-3 recommendations
        ranked, score = agent.select_arm(x)
        
        # Simulate user interaction
        chosen_arm = random.choice(ranked)
        rating_sim = simulate_reward(user, places[chosen_arm])
        rating_norm = (rating_sim - 1) / 4
        reward = 1 + rating_norm
        
        # Update chosen arm
        agent.update(chosen_arm, x, reward)
        user_rewards[user["id"]].append(reward)
        arm_history.append(chosen_arm)
        
        # Penalize non-chosen arms
        for arm in ranked:
            if arm != chosen_arm:
                agent.update(arm, x, -0.3)
                user_rewards[user["id"]].append(-0.3)
                arm_history.append(arm)
        
        avg_rewards.append(np.mean([r for lst in user_rewards.values() for r in lst]))
        
        if verbose and (round_num + 1) % 500 == 0:
            print(f"Round {round_num + 1}/{n_rounds} - Avg Reward: {np.mean(avg_rewards[-100:]):.3f}")
    
    final_avg = np.mean(avg_rewards[-100:])
    if verbose:
        print(f"✓ Training complete! Final avg reward: {final_avg:.3f}")
    
    return agent, user_rewards, avg_rewards, arm_history


def train_thompson_sampling(user_profiles, places, feature_columns, n_rounds=1000, verbose=True):
    """Train Thompson Sampling agent."""
    if verbose:
        print("\n" + "=" * 60)
        print("Training Thompson Sampling Agent")
        print("=" * 60)
    
    agent = ContextualThompsonSampling(n_arms=len(places), d=len(feature_columns), alpha=0.1)
    user_rewards = {u["id"]: [] for u in user_profiles}
    avg_rewards = []
    arm_history = []
    
    for round_num in range(n_rounds):
        user = random.choice(user_profiles)
        x = context_to_vector(user, feature_columns)
        
        # Select top-3 recommendations
        ranked, score = agent.select_arm(x)
        
        # Simulate user interaction
        chosen_arm = random.choice(ranked)
        rating_sim = simulate_reward(user, places[chosen_arm])
        rating_norm = (rating_sim - 1) / 4
        reward = 1 + rating_norm
        
        # Update chosen arm
        agent.update(chosen_arm, x, reward)
        user_rewards[user["id"]].append(reward)
        arm_history.append(chosen_arm)
        
        # Penalize non-chosen arms
        for arm in ranked:
            if arm != chosen_arm:
                agent.update(arm, x, -0.3)
                user_rewards[user["id"]].append(-0.3)
                arm_history.append(arm)
        
        avg_rewards.append(np.mean([r for lst in user_rewards.values() for r in lst]))
        
        if verbose and (round_num + 1) % 500 == 0:
            print(f"Round {round_num + 1}/{n_rounds} - Avg Reward: {np.mean(avg_rewards[-100:]):.3f}")
    
    final_avg = np.mean(avg_rewards[-100:])
    if verbose:
        print(f"✓ Training complete! Final avg reward: {final_avg:.3f}")
    
    return agent, user_rewards, avg_rewards, arm_history


def save_models(agent_eg, agent_linucb, agent_ts, output_dir='saved_models'):
    """Save trained models to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'epsilon_greedy.pkl'), 'wb') as f:
        pickle.dump(agent_eg, f)
    
    with open(os.path.join(output_dir, 'linucb.pkl'), 'wb') as f:
        pickle.dump(agent_linucb, f)
    
    with open(os.path.join(output_dir, 'thompson_sampling.pkl'), 'wb') as f:
        pickle.dump(agent_ts, f)
    
    print(f"\n✅ Models saved to {output_dir}/")
    print(f"  - epsilon_greedy.pkl")
    print(f"  - linucb.pkl")
    print(f"  - thompson_sampling.pkl")


def plot_comparison(avg_rewards_eg, avg_rewards_linucb, avg_rewards_ts, save_path=None):
    """Plot comparison of all three models."""
    plt.figure(figsize=(14, 6))
    
    # Smooth the curves with moving average
    window = 50
    
    def smooth(data, window):
        return pd.Series(data).rolling(window=window, min_periods=1).mean()
    
    plt.plot(smooth(avg_rewards_eg, window), label='Epsilon-Greedy', linewidth=2, alpha=0.8)
    plt.plot(smooth(avg_rewards_linucb, window), label='LinUCB', linewidth=2, alpha=0.8)
    plt.plot(smooth(avg_rewards_ts, window), label='Thompson Sampling', linewidth=2, alpha=0.8)
    
    plt.xlabel('Training Rounds', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Learning Curves: Average Reward Over Time (Smoothed)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def print_summary(avg_rewards_eg, avg_rewards_linucb, avg_rewards_ts):
    """Print training summary."""
    models = ['Epsilon-Greedy', 'LinUCB', 'Thompson Sampling']
    
    final_rewards = [
        np.mean(avg_rewards_eg[-100:]),
        np.mean(avg_rewards_linucb[-100:]),
        np.mean(avg_rewards_ts[-100:])
    ]
    
    total_rewards = [
        np.sum(avg_rewards_eg),
        np.sum(avg_rewards_linucb),
        np.sum(avg_rewards_ts)
    ]
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} | {'Final Avg Reward':>17} | {'Total Reward':>12}")
    print("-" * 70)
    for i, model in enumerate(models):
        print(f"{model:<20} | {final_rewards[i]:>17.3f} | {total_rewards[i]:>12.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Train RL models for travel recommendation')
    parser.add_argument('--rounds', type=int, default=2000, help='Number of training rounds (default: 2000)')
    parser.add_argument('--save', action='store_true', help='Save trained models')
    parser.add_argument('--plot', action='store_true', help='Display comparison plot')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ADAPTIVE TRAVEL RECOMMENDATION - MODEL TRAINING")
    print("=" * 70)
    print(f"Training rounds: {args.rounds}")
    print("=" * 70)
    
    # Load data
    user_profiles, places, feature_columns = load_data()
    print(f"✓ Data loaded: {len(user_profiles)} users, {len(places)} places, {len(feature_columns)} features")
    
    # Train all models
    agent_eg, user_rewards_eg, avg_rewards_eg, arm_history_eg = train_epsilon_greedy(
        user_profiles, places, n_rounds=args.rounds
    )
    
    agent_linucb, user_rewards_linucb, avg_rewards_linucb, arm_history_linucb = train_linucb(
        user_profiles, places, feature_columns, n_rounds=args.rounds
    )
    
    agent_ts, user_rewards_ts, avg_rewards_ts, arm_history_ts = train_thompson_sampling(
        user_profiles, places, feature_columns, n_rounds=args.rounds
    )
    
    # Print summary
    print_summary(avg_rewards_eg, avg_rewards_linucb, avg_rewards_ts)
    
    # Save models if requested
    if args.save:
        save_models(agent_eg, agent_linucb, agent_ts)
    
    # Plot comparison
    if args.plot or args.save_plot:
        plot_comparison(avg_rewards_eg, avg_rewards_linucb, avg_rewards_ts, save_path=args.save_plot)


if __name__ == "__main__":
    main()
