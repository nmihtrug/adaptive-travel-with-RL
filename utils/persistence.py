"""
Persistence utilities for saving and loading models, data, and results.

This module provides functions to:
- Save/load trained agents
- Save/load training history
- Save/load datasets
- Export results to various formats
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple


def create_save_directory(base_dir: str = "saved_models") -> str:
    """
    Create a timestamped directory for saving models and results.
    
    Args:
        base_dir (str): Base directory name
        
    Returns:
        str: Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_agent(agent, agent_name: str, save_dir: str) -> str:
    """
    Save a trained agent to disk.
    
    Args:
        agent: Trained agent object
        agent_name (str): Name of the agent (e.g., 'epsilon_greedy', 'linucb')
        save_dir (str): Directory to save to
        
    Returns:
        str: Path to saved file
    """
    filepath = os.path.join(save_dir, f"{agent_name}_agent.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)
    print(f"✅ Saved agent to: {filepath}")
    return filepath


def load_agent(filepath: str):
    """
    Load a trained agent from disk.
    
    Args:
        filepath (str): Path to saved agent file
        
    Returns:
        Loaded agent object
    """
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    print(f"✅ Loaded agent from: {filepath}")
    return agent


def save_training_history(
    history: Dict[str, Any],
    save_dir: str,
    filename: str = "training_history.pkl"
) -> str:
    """
    Save training history (rewards, arm selections, etc.).
    
    Args:
        history (Dict): Dictionary containing training history
        save_dir (str): Directory to save to
        filename (str): Filename for history file
        
    Returns:
        str: Path to saved file
    """
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"✅ Saved training history to: {filepath}")
    return filepath


def load_training_history(filepath: str) -> Dict[str, Any]:
    """
    Load training history from disk.
    
    Args:
        filepath (str): Path to history file
        
    Returns:
        Dict: Training history
    """
    with open(filepath, 'rb') as f:
        history = pickle.load(f)
    print(f"✅ Loaded training history from: {filepath}")
    return history


def save_metrics(metrics: Dict, save_dir: str, filename: str = "metrics.json") -> str:
    """
    Save evaluation metrics as JSON.
    
    Args:
        metrics (Dict): Metrics dictionary
        save_dir (str): Directory to save to
        filename (str): Filename for metrics file
        
    Returns:
        str: Path to saved file
    """
    filepath = os.path.join(save_dir, filename)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    metrics_clean = convert_numpy(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"✅ Saved metrics to: {filepath}")
    return filepath


def load_metrics(filepath: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        filepath (str): Path to metrics file
        
    Returns:
        Dict: Metrics dictionary
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    print(f"✅ Loaded metrics from: {filepath}")
    return metrics


def save_config(config: Dict, save_dir: str, filename: str = "config.json") -> str:
    """
    Save configuration/hyperparameters as JSON.
    
    Args:
        config (Dict): Configuration dictionary
        save_dir (str): Directory to save to
        filename (str): Filename for config file
        
    Returns:
        str: Path to saved file
    """
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved config to: {filepath}")
    return filepath


def save_dataset(
    places: List[Dict],
    user_profiles: List[Dict],
    keywords: List[str],
    save_dir: str,
    filename: str = "dataset.pkl"
) -> str:
    """
    Save dataset (places, users, keywords) to disk.
    
    Args:
        places (List[Dict]): List of places
        user_profiles (List[Dict]): List of user profiles
        keywords (List[str]): List of keywords
        save_dir (str): Directory to save to
        filename (str): Filename for dataset
        
    Returns:
        str: Path to saved file
    """
    dataset = {
        'places': places,
        'user_profiles': user_profiles,
        'keywords': keywords,
        'place_types': keywords  # Keep for backward compatibility
    }
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✅ Saved dataset to: {filepath}")
    return filepath


def load_dataset(filepath: str) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Load dataset from disk.
    
    Args:
        filepath (str): Path to dataset file (.pkl) or CSV file
        
    Returns:
        Tuple: (places, user_profiles, keywords)
    """
    # Handle CSV files
    if filepath.endswith('.csv'):
        from data.synthetic_data import load_places_from_csv
        places, keywords = load_places_from_csv(filepath)
        # Return empty user profiles since CSV only has places
        return places, [], keywords
    
    # Handle pickle files
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"✅ Loaded dataset from: {filepath}")
    
    # Support both 'keywords' and 'place_types' for backward compatibility
    keywords = dataset.get('keywords', dataset.get('place_types', []))
    
    return dataset['places'], dataset['user_profiles'], keywords


def export_results_to_csv(
    results: Dict,
    save_dir: str,
    filename: str = "results.csv"
) -> str:
    """
    Export comparison results to CSV.
    
    Args:
        results (Dict): Results dictionary from compare_agents
        save_dir (str): Directory to save to
        filename (str): Filename for CSV
        
    Returns:
        str: Path to saved file
    """
    # Convert results to DataFrame
    data = []
    for agent_name, metrics in results.items():
        row = {'agent': agent_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Exported results to: {filepath}")
    return filepath


def save_complete_experiment(
    agents: Dict,
    training_histories: Dict,
    metrics: Dict,
    config: Dict,
    places: List[Dict],
    user_profiles: List[Dict],
    keywords: List[str],
    base_dir: str = "saved_models"
) -> str:
    """
    Save a complete experiment including agents, history, metrics, and data.
    
    Args:
        agents (Dict): Dictionary of {name: (agent, type)} pairs
        training_histories (Dict): Dictionary of training histories per agent
        metrics (Dict): Evaluation metrics
        config (Dict): Experiment configuration
        places (List[Dict]): Places dataset
        user_profiles (List[Dict]): User profiles
        keywords (List[str]): Keywords/place types
        base_dir (str): Base directory for saving
        
    Returns:
        str: Path to save directory
    """
    # Create save directory
    save_dir = create_save_directory(base_dir)
    
    print(f"\n💾 Saving complete experiment to: {save_dir}")
    print("=" * 60)
    
    # Save agents
    print("\n📦 Saving agents...")
    for name, (agent, _) in agents.items():
        agent_filename = name.lower().replace(' ', '_').replace('-', '_')
        save_agent(agent, agent_filename, save_dir)
    
    # Save training histories
    print("\n📊 Saving training histories...")
    save_training_history(training_histories, save_dir, "training_histories.pkl")
    
    # Save metrics
    print("\n📈 Saving metrics...")
    save_metrics(metrics, save_dir)
    export_results_to_csv(metrics, save_dir)
    
    # Save config
    print("\n⚙️  Saving configuration...")
    save_config(config, save_dir)
    
    # Save dataset
    print("\n📋 Saving dataset...")
    save_dataset(places, user_profiles, keywords, save_dir)
    
    # Create README in save directory
    readme_path = os.path.join(save_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"Experiment saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Contents:\n")
        f.write("  - *_agent.pkl: Trained agent models\n")
        f.write("  - training_histories.pkl: Training histories for all agents\n")
        f.write("  - metrics.json: Evaluation metrics\n")
        f.write("  - results.csv: Results in CSV format\n")
        f.write("  - config.json: Experiment configuration\n")
        f.write("  - dataset.pkl: Dataset used for training\n\n")
        f.write("To load:\n")
        f.write("  from utils.persistence import load_agent, load_metrics\n")
        f.write("  agent = load_agent('epsilon_greedy_agent.pkl')\n")
        f.write("  metrics = load_metrics('metrics.json')\n")
    
    print(f"\n✅ Complete experiment saved to: {save_dir}")
    print("=" * 60)
    
    return save_dir


def load_experiment(experiment_dir: str) -> Dict[str, Any]:
    """
    Load a complete saved experiment.
    
    Args:
        experiment_dir (str): Path to experiment directory
        
    Returns:
        Dict: Dictionary containing all loaded components
    """
    print(f"\n📂 Loading experiment from: {experiment_dir}")
    print("=" * 60)
    
    experiment = {}
    
    # Load agents
    print("\n📦 Loading agents...")
    agents = {}
    for filename in os.listdir(experiment_dir):
        if filename.endswith('_agent.pkl'):
            agent_name = filename.replace('_agent.pkl', '')
            filepath = os.path.join(experiment_dir, filename)
            agents[agent_name] = load_agent(filepath)
    experiment['agents'] = agents
    
    # Load training histories
    history_path = os.path.join(experiment_dir, 'training_histories.pkl')
    if os.path.exists(history_path):
        print("\n📊 Loading training histories...")
        experiment['training_histories'] = load_training_history(history_path)
    
    # Load metrics
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print("\n📈 Loading metrics...")
        experiment['metrics'] = load_metrics(metrics_path)
    
    # Load config
    config_path = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_path):
        print("\n⚙️  Loading configuration...")
        with open(config_path, 'r') as f:
            experiment['config'] = json.load(f)
        print(f"✅ Loaded config from: {config_path}")
    
    # Load dataset
    dataset_path = os.path.join(experiment_dir, 'dataset.pkl')
    if os.path.exists(dataset_path):
        print("\n📋 Loading dataset...")
        places, users, keywords = load_dataset(dataset_path)
        experiment['places'] = places
        experiment['user_profiles'] = users
        experiment['keywords'] = keywords
        experiment['place_types'] = keywords  # Backward compatibility
    
    print(f"\n✅ Experiment loaded successfully!")
    print("=" * 60)
    
    return experiment
