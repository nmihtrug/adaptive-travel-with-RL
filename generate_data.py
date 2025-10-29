"""
Data generation script - Creates and saves synthetic datasets.

This script generates synthetic travel recommendation datasets and saves them
for later use in training. This separates data generation from model training.

Usage:
    python generate_data.py --output data/
    python generate_data.py --n-users 100 --n-interactions 5000 --output data/
"""

import os
import argparse
import numpy as np
import pandas as pd
import random
from datetime import datetime
from typing import List, Dict, Tuple

def load_places_from_csv(csv_path: str) -> Tuple[List[Dict], List[str]]:
    """
    Load places from CSV file with columns: Location Name, Keywords, Rating.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (places, keywords) where:
        - places: List of place dictionaries with name, keywords, rating
        - keywords: List of unique keywords found across all places
    """
    df = pd.read_csv(csv_path)
    
    # Extract unique keywords
    all_keywords = set()
    places = []
    
    for idx, row in df.iterrows():
        # Parse keywords (comma-separated string)
        keywords_str = str(row['Keywords'])
        keywords = [k.strip() for k in keywords_str.split(',')]
        
        place = {
            'name': row['Location Name'],
            'keywords': keywords,
            'rating': float(row['Rating'])
        }
        places.append(place)
        all_keywords.update(keywords)
    
    keywords_list = sorted(list(all_keywords))
    
    return places, keywords_list


def generate_user_profiles(n_users: int = 10, keywords: List[str] = None, preference_diversity: float = 0.3) -> List[Dict]:
    """
    Generate synthetic user profiles with hobby/keyword preferences.
    
    Args:
        n_users: Number of user profiles to generate
        keywords: List of keywords to use for hobbies
        preference_diversity: Higher values create more diverse preferences
        
    Returns:
        List of user profile dictionaries with id and hobbies
    """
    if keywords is None:
        keywords = ['biển', 'chùa', 'lịch sử', 'văn hóa', 'thiên nhiên']
    
    user_profiles = []
    for i in range(n_users):
        # Generate Dirichlet distribution for preferences
        if preference_diversity < 0.5:
            # More specialized users
            prefs = np.random.dirichlet([1] * len(keywords))
        else:
            # More diverse users
            prefs = np.random.dirichlet([3] * len(keywords))
        
        hobbies_dict = {keyword: float(pref) for keyword, pref in zip(keywords, prefs)}
        
        user_profiles.append({
            "id": i,
            "hobbies": hobbies_dict,
        })
    
    return user_profiles


def simulate_reward(user: Dict, place: Dict, noise_std: float = 0.05) -> int:
    """
    Simulate user's reward (click/rating) for a recommended place.
    Uses keyword/hobby matching between user and place.
    
    Args:
        user: User profile with hobbies dict
        place: Place with keywords list and rating
        noise_std: Standard deviation of noise
        
    Returns:
        1 if user clicks, 0 otherwise
    """
    # Calculate preference score based on matching keywords
    user_hobbies = user.get("hobbies", user.get("prefs", {}))
    place_keywords = place.get("keywords", [])
    
    # Sum up user's preference weights for matching keywords
    pref_score = sum(user_hobbies.get(kw, 0) for kw in place_keywords)
    
    # Normalize by number of keywords and incorporate place rating
    if len(place_keywords) > 0:
        pref_score = pref_score / len(place_keywords)
    
    # Incorporate place rating (0.0 to 1.0)
    place_rating = place.get('rating', 0.5)
    
    # Combined probability: weighted average of preference and rating
    prob = 0.7 * pref_score + 0.3 * place_rating
    
    # Add noise
    prob = prob + np.random.normal(0, noise_std)
    prob = np.clip(prob, 0, 1)
    
    return 1 if np.random.rand() < prob else 0


def context_to_vector(user: Dict, keywords: List[str]) -> np.ndarray:
    """
    Convert user hobby/preference dictionary to feature vector.
    
    Args:
        user: User profile with hobbies dict
        keywords: List of keywords in specific order
        
    Returns:
        Numpy array of user's preference weights for each keyword
    """
    user_hobbies = user.get("hobbies", user.get("prefs", {}))
    return np.array([user_hobbies.get(kw, 0) for kw in keywords])


def generate_interaction_data(
    places: List[Dict], 
    user_profiles: List[Dict], 
    n_interactions: int = 1000, 
    noise_std: float = 0.05
) -> List[Dict]:
    """
    Generate synthetic interaction data (user-place-reward triplets).
    
    Args:
        places: List of place dictionaries
        user_profiles: List of user profile dictionaries
        n_interactions: Number of interactions to generate
        noise_std: Standard deviation of noise in reward simulation
        
    Returns:
        List of interaction dictionaries
    """
    interactions = []
    for _ in range(n_interactions):
        user = random.choice(user_profiles)
        place_idx = random.randint(0, len(places) - 1)
        place = places[place_idx]
        clicked = simulate_reward(user, place, noise_std)
        
        interactions.append({
            'user_id': user['id'],
            'place_idx': place_idx,
            'place_name': place['name'],
            'place_keywords': ', '.join(place['keywords']),
            'place_rating': place['rating'],
            'clicked': clicked
        })
    return interactions


def save_user_profiles_to_csv(user_profiles: List[Dict], output_path: str):
    """Save user profiles to CSV file."""
    if not user_profiles:
        return
    
    # Get all keywords
    all_keywords = set()
    for user in user_profiles:
        all_keywords.update(user['hobbies'].keys())
    
    keywords_sorted = sorted(list(all_keywords))
    
    # Create DataFrame
    data = []
    for user in user_profiles:
        row = {'user_id': user['id']}
        for kw in keywords_sorted:
            row[kw] = user['hobbies'].get(kw, 0.0)
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(user_profiles)} user profiles to {output_path}")


def save_interactions_to_csv(interactions: List[Dict], output_path: str):
    """Save interactions to CSV file."""
    df = pd.DataFrame(interactions)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(interactions)} interactions to {output_path}")


def generate_dataset(
    dataset_path: str,
    n_users: int,
    output_dir: str,
    preference_diversity: float = 0.3
):
    """Generate user profiles and interaction data from places dataset."""
    
    print(f"\n📂 Loading places from: {dataset_path}")
    places, keywords = load_places_from_csv(dataset_path)
    
    print(f"✅ Loaded {len(places)} places with {len(keywords)} unique keywords")
    print(f"   Keywords: {keywords[:10]}...")
    
    # Generate user profiles
    print(f"\n Generating {n_users} user profiles...")
    user_profiles = generate_user_profiles(
        n_users=n_users,
        keywords=keywords,
        preference_diversity=preference_diversity
    )
    
    print(f"✅ Generated {len(user_profiles)} user profiles")
    
    # Save to CSV files
    os.makedirs(output_dir, exist_ok=True)
    
    user_profiles_path = os.path.join(output_dir, 'user_profiles.csv')
    
    save_user_profiles_to_csv(user_profiles, user_profiles_path)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   User profiles: {user_profiles_path}")
    
    return user_profiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic travel recommendation datasets"
    )
    
    parser.add_argument(
        '--dataset', type=str, default="./data/final_dataset.csv",
        help='Path to places dataset CSV file'
    )
    parser.add_argument(
        '--n-users', type=int, default=10,
        help='Number of users to generate'
    )
    parser.add_argument(
        '--diversity', type=float, default=0.3,
        help='User preference diversity (0-1, lower=specialized, higher=diverse)'
    )
    parser.add_argument(
        '--output', type=str, default='./data/',
        help='Output directory for generated CSV files'
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        dataset_path=args.dataset,
        n_users=args.n_users,
        output_dir=args.output,
        preference_diversity=args.diversity
    )
    


if __name__ == "__main__":
    main()
