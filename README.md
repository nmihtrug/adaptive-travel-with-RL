# Adaptive Travel Recommendations with Reinforcement Learning

A comprehensive implementation of multi-armed bandit algorithms for personalized travel recommendations, featuring Epsilon-Greedy, LinUCB, and Thompson Sampling approaches.

## 🎯 Project Overview

This project implements and compares three reinforcement learning algorithms for adaptive travel recommendations based on Vietnamese travel destinations:

- **Epsilon-Greedy**: Simple exploration-exploitation strategy
- **LinUCB**: Contextual bandit with upper confidence bound
- **Thompson Sampling**: Bayesian approach with posterior sampling

The system uses real Vietnamese travel destinations from `data/final_dataset.csv` and generates synthetic user click behavior to train recommendation models.

## 📁 Project Structure

```
adaptive-travel-with-RL/
├── models/
│   ├── __init__.py
│   └── agents.py              # RL agent implementations
├── data/
│   ├── __init__.py
│   ├── synthetic_data.py      # User behavior generation
│   └── final_dataset.csv      # Vietnamese travel destinations (Location, Keywords, Rating)
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # Plotting functions
│   ├── persistence.py         # Save/load models and data
│   └── interactive.py         # Interactive recommendation interfaces
├── saved_models/              # Trained models and results
├── notebooks/                 # Jupyter notebooks for analysis
├── generate_data.py           # Generate random user click behavior
├── train.py                   # Train RL models
├── load_models.py             # Load and use trained models
├── requirements.txt           # Dependencies
└── README.md
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/nmihtrug/adaptive-travel-with-RL.git
cd adaptive-travel-with-RL

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Two-Step Pipeline

**Step 1: Generate Random User Click Behavior**

```bash
# Generate default dataset (20 users, 2000 interactions)
python generate_data.py --output data/user_clicks.pkl

# Generate custom dataset
python generate_data.py \
    --n-users 50 \
    --n-interactions 5000 \
    --diversity 0.7 \
    --output data/large_clicks.pkl

# Options:
#   --n-users: Number of users to simulate (default: 20)
#   --n-interactions: Number of click interactions (default: 2000)
#   --diversity: Preference diversity 0-1 (default: 0.5, lower=specialized, higher=diverse)
#   --csv: Path to CSV with places (default: data/final_dataset.csv)
#   --output: Output path for dataset (default: data/user_clicks.pkl)
```

**Step 2: Train Models**

```bash
# Train all models (Epsilon-Greedy, LinUCB, Thompson Sampling)
python train.py --dataset data/user_clicks.pkl

# Train specific models
python train.py --dataset data/user_clicks.pkl --agents linucb thompson

# Custom training parameters
python train.py \
    --dataset data/user_clicks.pkl \
    --n-rounds 3000 \
    --top-k 5 \
    --epsilon 0.1 \
    --alpha-linucb 0.2

# Options:
#   --dataset: Path to dataset file (required)
#   --agents: Which agents to train (default: all)
#   --n-rounds: Number of training rounds (default: 2000)
#   --top-k: Number of recommendations per round (default: 3)
#   --epsilon: Exploration rate for Epsilon-Greedy (default: 0.2)
#   --alpha-linucb: Exploration parameter for LinUCB (default: 0.1)
#   --alpha-ts: Prior std for Thompson Sampling (default: 0.1)
```

### Data Format

The `data/final_dataset.csv` contains Vietnamese travel destinations with:
- **Location Name**: Name of the place (e.g., "Đảo Phú Quốc")
- **Keywords**: Comma-separated categories (e.g., "biển, chụp ảnh, nghĩ dưỡng, vui chơi")
- **Rating**: Base quality rating (0-1 scale)

User profiles are generated with preferences over these keywords, and click behavior is simulated based on keyword matching and ratings.

### Interactive Mode

```python
from utils.persistence import load_experiment

# Load a trained experiment
experiment = load_experiment('saved_models/run_20251029_143932')
places = experiment['places']
agents = experiment['agents']

# Get recommendations for a user
user_prefs = {'biển': 0.8, 'vui chơi': 0.6, 'chụp ảnh': 0.4}
# ... use agent to get recommendations ...
```

### Generate Synthetic User Behavior

```python
from data.synthetic_data import (
    load_places_from_csv,
    generate_user_profiles,
    generate_interaction_data
)

# Load places from CSV
places, keywords = load_places_from_csv('data/final_dataset.csv')

# Generate user profiles with keyword preferences
users = generate_user_profiles(n_users=20, keywords=keywords, preference_diversity=0.5)

# Generate click interactions
interactions = generate_interaction_data(
    places=places,
    user_profiles=users,
    keywords=keywords,
    n_interactions=2000
)
```

### Save and Load Models

```python
from utils.persistence import load_experiment, save_complete_experiment

# Load complete experiment
experiment = load_experiment('saved_models/run_20251029_120000')
agent = experiment['agents']['LinUCB']
metrics = experiment['metrics']
places = experiment['places']

# Use interactive loader
python load_models.py
```

## 📊 Algorithms

### 1. Epsilon-Greedy
- **Exploration**: Random selection with probability ε
- **Exploitation**: Select best-performing arm with probability 1-ε
- **Best for**: Simple, fast, non-contextual scenarios

### 2. LinUCB (Linear Upper Confidence Bound)
- **Contextual**: Uses user preferences as features
- **UCB**: Balances exploitation and exploration via confidence bounds
- **Best for**: Contextual bandits with linear reward structure

### 3. Thompson Sampling
- **Bayesian**: Maintains posterior distributions over parameters
- **Sampling**: Draws from posterior to balance exploration/exploitation
- **Best for**: Complex scenarios with uncertainty quantification

## 📈 Performance Metrics

The models are evaluated on:
- **Average Reward**: Overall recommendation quality
- **User-specific Rewards**: Personalization effectiveness
- **Convergence Speed**: Learning efficiency
- **Click-Through Rate**: User engagement

## 🎮 Features

- **Real Vietnamese Data**: 150+ Vietnamese travel destinations
- **Keyword-Based Matching**: Multi-dimensional user preferences
- **Modular Design**: Easy to extend and customize
- **Synthetic User Behavior**: Generate diverse test scenarios
- **Comprehensive Visualization**: Learning curves, distributions, comparisons
- **Multiple Algorithms**: Compare different RL approaches
- **Separated Pipeline**: Data generation and training are independent

## 📝 Example Results

Each algorithm learns user preferences over time:
- Users preferring **biển** (beach) → Đảo Phú Quốc, Bãi Dài Cam Ranh
- Users preferring **văn hóa** (culture) → Tháp Bà Ponagar, Thành cổ Diên Khánh
- Users preferring **vui chơi** (entertainment) → VinWonders Nha Trang

## 🔬 Experimentation

Use the Jupyter notebooks for:
- **synthetic_data.ipynb**: Experiment with data generation
- **analysis.ipynb**: Compare algorithm performance
- **main_eGreedy.ipynb**: Original implementation reference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

nmihtrug

## 🙏 Acknowledgments

- Based on multi-armed bandit theory
- Inspired by real-world recommendation systems
- Vietnamese travel destinations dataset
