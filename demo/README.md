# Adaptive Travel Recommendation Demo

This folder contains interactive Streamlit demos for the Adaptive Travel Recommendation System using Reinforcement Learning.

## 📁 Files

- `demo.py` - Main comprehensive demo with all three algorithms
- `interactive_recommendation_egreedy.py` - Epsilon-Greedy specific demo
- `interactive_recommendation_linucb.py` - LinUCB specific demo
- `interactive_recommendation_ts.py` - Thompson Sampling specific demo

## 🚀 How to Run

### Prerequisites

1. Make sure you have trained the models first:
   ```bash
   # From the project root, run the training notebook
   jupyter notebook notebooks/train.ipynb
   ```

2. Install required dependencies:
   ```bash
   pip install streamlit
   ```

### Running the Demos

#### Main Demo (Recommended)
The main demo allows you to switch between all three algorithms:

```bash
cd demo
streamlit run demo.py
```

#### Individual Algorithm Demos

**Epsilon-Greedy:**
```bash
cd demo
streamlit run interactive_recommendation_egreedy.py
```

**LinUCB:**
```bash
cd demo
streamlit run interactive_recommendation_linucb.py
```

**Thompson Sampling:**
```bash
cd demo
streamlit run interactive_recommendation_ts.py
```

## 🎮 Features

### Main Demo (`demo.py`)
- **Switch between algorithms** on the fly
- **Select user profiles** (for contextual algorithms)
- **Adjust number of recommendations** (5-20)
- **Real-time feedback** and model updates
- **Save updated models** back to disk
- **View statistics** and performance metrics

### Individual Demos
Each specialized demo provides:
- **Top 10 recommendations** by default
- **Interactive rating system** (0-1 scale)
- **Real-time model learning** from feedback
- **User preference visualization** (for contextual models)
- **Algorithm-specific metrics** and information

## 📊 How It Works

1. **Load Data**: The app loads 147 Vietnamese travel destinations and 20 user profiles
2. **Select Configuration**: Choose algorithm and (optionally) user profile
3. **Get Recommendations**: View top-K personalized travel recommendations
4. **Provide Feedback**: Rate destinations you're interested in
5. **See Learning**: Watch the model adapt to your preferences in real-time

## 🎯 Algorithms

### Epsilon-Greedy
- Non-contextual bandit
- Explores random options with probability ε
- Learns which destinations are generally popular
- Fast and simple

### LinUCB (Contextual Bandit)
- Considers user preferences
- Uses confidence bounds for exploration
- Provides personalized recommendations
- Balances exploration vs exploitation

### Thompson Sampling
- Bayesian approach to contextual bandits
- Samples from posterior distributions
- Natural exploration through probability matching
- Highly effective for personalization

## 💡 Tips

- **For Epsilon-Greedy**: Provide consistent feedback to help it learn popular destinations
- **For LinUCB/TS**: Try different user profiles to see personalization in action
- **For best results**: Rate multiple destinations to help the model learn faster
- **Compare algorithms**: Try the same user profile with different algorithms to see how they differ

## 📈 Understanding the Output

- **AI Score**: The algorithm's confidence in recommending this destination
- **Match %** (contextual only): How well the destination matches your preferences
- **Rating**: The actual rating from the dataset
- **Type**: The primary category of the destination
- **Keywords**: Additional descriptive tags

## 🔧 Troubleshooting

**Error: "Pre-trained model not found"**
- Run the training notebook first (`notebooks/train.ipynb`)
- Make sure models are saved in `saved_models/` directory

**Error: "No module named 'streamlit'"**
- Install Streamlit: `pip install streamlit`

**App not updating after feedback**
- Click the "Refresh Recommendations" button
- The app should auto-refresh after each feedback submission

## 📝 Notes

- Models are loaded once when the app starts (cached for performance)
- Feedback updates the model in memory
- Use "Save Current Model" button to persist changes to disk
- Each algorithm maintains its own feedback counter
