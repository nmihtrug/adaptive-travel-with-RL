"""
Interactive recommendation interfaces for real-time user feedback.

Provides interactive CLI interfaces for:
- Epsilon-Greedy recommendations
- LinUCB recommendations
- Thompson Sampling recommendations
"""

import numpy as np
from typing import List, Dict
from IPython.display import clear_output
from data.synthetic_data import context_to_vector


def interactive_recommendation_egreedy(
    agent,
    places: List[Dict],
    top_k: int = 3
):
    """
    Interactive recommendation system using Epsilon-Greedy agent.
    
    Args:
        agent: Trained EpsilonGreedyAgent instance
        places (List[Dict]): List of places
        top_k (int): Number of recommendations to show
    """
    clear_output(wait=True)
    print("=" * 60)
    print("🌍 Welcome to the Travel Recommender (Epsilon-Greedy)!")
    print("=" * 60)
    
    while True:
        # Get top-k recommendations
        ranked = np.argsort(agent.values)[::-1][:top_k]
        
        print("\n📍 Top Recommendations for You:")
        print("-" * 60)
        for i, idx in enumerate(ranked):
            place = places[idx]
            print(f"   {i+1}. {place['name']:20s} ({place['type']:12s}) | Score: {agent.values[idx]:.3f}")
        
        # Get user input
        print("\n" + "-" * 60)
        clicks = input("👆 Click on recommendations (e.g., '1 3') or 'q' to quit: ")
        
        if clicks.lower() == 'q':
            print("\n👋 Thank you for using the Travel Recommender! Goodbye!")
            break

        # Parse clicked arms
        clicked_arms = []
        for c in clicks.split():
            try:
                idx = int(c) - 1
                if 0 <= idx < len(ranked):
                    clicked_arms.append(ranked[idx])
            except:
                continue

        # Update based on feedback
        for arm in ranked:
            if arm in clicked_arms:
                try:
                    rating = float(input(f"⭐ Rate {places[arm]['name']} (0-1): "))
                    rating = np.clip(rating, 0, 1)
                except:
                    rating = 1.0  # Default to positive
                agent.update(arm, rating)
            else:
                # Penalize non-clicked recommendations
                agent.update(arm, -0.1)

        print("\n✅ Model updated with your feedback!")
        print("=" * 60)


def interactive_recommendation_linucb(
    agent,
    places: List[Dict],
    user_profiles: List[Dict],
    place_types: List[str],
    top_k: int = 3
):
    """
    Interactive recommendation system using LinUCB agent.
    
    Args:
        agent: Trained LinUCBAgent instance
        places (List[Dict]): List of places
        user_profiles (List[Dict]): List of user profiles
        place_types (List[str]): List of place types
        top_k (int): Number of recommendations to show
    """
    clear_output(wait=True)
    print("=" * 60)
    print("🌍 Welcome to the Travel Recommender (LinUCB)!")
    print("=" * 60)

    while True:
        print("\n👤 Select your user profile:")
        for user in user_profiles:
            prefs_str = ", ".join([f"{k}: {v:.2f}" for k, v in user['prefs'].items()])
            print(f"   User {user['id']}: {prefs_str}")
        
        user_input = input("\nEnter user ID (0-{}) or 'q' to quit: ".format(len(user_profiles)-1))
        
        if user_input.lower() == 'q':
            print("\n👋 Thank you for using the Travel Recommender! Goodbye!")
            break

        try:
            user_id = int(user_input)
            if user_id < 0 or user_id >= len(user_profiles):
                print("❌ Invalid user ID! Please try again.")
                continue
        except:
            print("❌ Invalid input! Please try again.")
            continue

        user = user_profiles[user_id]
        x = context_to_vector(user, place_types)

        # Compute UCB scores
        scores = []
        for arm in range(len(places)):
            A_inv = np.linalg.inv(agent.A[arm])
            theta = A_inv @ agent.b[arm]
            score = theta @ x + agent.alpha * np.sqrt(x @ A_inv @ x)
            scores.append(score)

        # Get top-k recommendations
        ranked = np.argsort(scores)[::-1][:top_k]

        # Display recommendations
        print(f"\n📍 Top Recommendations for User {user_id}:")
        print("-" * 60)
        for i, idx in enumerate(ranked):
            place = places[idx]
            print(f"   {i+1}. {place['name']:20s} ({place['type']:12s}) | Score: {scores[idx]:.3f}")

        # Get user feedback
        print("\n" + "-" * 60)
        clicks = input("👆 Click on recommendations (e.g., '1 3') or 'q' to quit: ")
        
        if clicks.lower() == 'q':
            print("\n👋 Thank you for using the Travel Recommender! Goodbye!")
            break

        # Parse clicked arms
        clicked_arms = []
        for c in clicks.split():
            try:
                idx = int(c) - 1
                if 0 <= idx < len(ranked):
                    clicked_arms.append(ranked[idx])
            except:
                continue

        # Update based on feedback
        for arm in ranked:
            if arm in clicked_arms:
                try:
                    rating = float(input(f"⭐ Rate {places[arm]['name']} (0-1): "))
                    rating = np.clip(rating, 0, 1)
                except:
                    rating = 1.0
                agent.update(arm, x, rating)
            else:
                agent.update(arm, x, -0.1)

        print("\n✅ Model updated with your feedback!")
        print("=" * 60)


def interactive_recommendation_thompson(
    agent,
    places: List[Dict],
    user_profiles: List[Dict],
    place_types: List[str],
    top_k: int = 3
):
    """
    Interactive recommendation system using Thompson Sampling agent.
    
    Args:
        agent: Trained ContextualThompsonSampling instance
        places (List[Dict]): List of places
        user_profiles (List[Dict]): List of user profiles
        place_types (List[str]): List of place types
        top_k (int): Number of recommendations to show
    """
    clear_output(wait=True)
    print("=" * 60)
    print("🌍 Welcome to the Travel Recommender (Thompson Sampling)!")
    print("=" * 60)
    
    while True:
        print("\n👤 Select your user profile:")
        for user in user_profiles:
            prefs_str = ", ".join([f"{k}: {v:.2f}" for k, v in user['prefs'].items()])
            print(f"   User {user['id']}: {prefs_str}")
        
        user_input = input("\nEnter user ID (0-{}) or 'q' to quit: ".format(len(user_profiles)-1))
        
        if user_input.lower() == 'q':
            print("\n👋 Thank you for using the Travel Recommender! Goodbye!")
            break

        try:
            user_id = int(user_input)
            if user_id < 0 or user_id >= len(user_profiles):
                print("❌ Invalid user ID! Please try again.")
                continue
        except:
            print("❌ Invalid input! Please try again.")
            continue

        user = user_profiles[user_id]
        x = context_to_vector(user, place_types)

        # Sample from posterior for each arm
        sampled_rewards = []
        for arm in range(agent.n_arms):
            A_inv = np.linalg.inv(agent.A[arm])
            mu = A_inv @ agent.b[arm]
            cov = agent.alpha**2 * A_inv
            theta_sample = np.random.multivariate_normal(mu, cov)
            sampled_rewards.append(x @ theta_sample)

        # Get top-k recommendations
        ranked = np.argsort(sampled_rewards)[::-1][:top_k]

        # Display recommendations
        print(f"\n📍 Top Recommendations for User {user_id}:")
        print("-" * 60)
        for i, idx in enumerate(ranked):
            place = places[idx]
            print(f"   {i+1}. {place['name']:20s} ({place['type']:12s})")

        # Get user feedback
        print("\n" + "-" * 60)
        clicks = input("👆 Click on recommendations (e.g., '1 2 3') or 'q' to quit: ")
        
        if clicks.lower() == 'q':
            print("\n👋 Thank you for using the Travel Recommender! Goodbye!")
            break

        # Parse clicked arms
        clicked_arms = []
        for c in clicks.split():
            try:
                idx = int(c) - 1
                if 0 <= idx < len(ranked):
                    clicked_arms.append(ranked[idx])
            except:
                continue

        # Update based on feedback
        for arm in ranked:
            if arm in clicked_arms:
                try:
                    rating = float(input(f"⭐ Rate {places[arm]['name']} (0-1): "))
                    rating = np.clip(rating, 0, 1)
                except:
                    rating = 1.0
                agent.update(arm, x, rating)
            else:
                agent.update(arm, x, 0.0)

        print("\n✅ Model updated with your feedback!")
        print("=" * 60)
