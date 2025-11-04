"""Interactive Epsilon-Greedy Recommendation Demo."""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('..')

from models.agents import EpsilonGreedyAgent


def load_data():
    """Load user profiles and places data."""
    # Load user profiles
    df_users = pd.read_csv('../data/gen/user_profiles.csv')
    feature_columns = [col for col in df_users.columns if col != 'user_id']
    
    user_profiles = []
    for _, row in df_users.iterrows():
        user_id = int(row['user_id'])
        prefs = {col: row[col] for col in feature_columns}
        user_profiles.append({"id": user_id, "prefs": prefs})
    
    # Load places
    df_places = pd.read_csv('../data/final_dataset.csv')
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
    
    return user_profiles, places, feature_columns


def load_model():
    """Load pre-trained Epsilon-Greedy model."""
    try:
        with open('../saved_models/epsilon_greedy.pkl', 'rb') as f:
            agent = pickle.load(f)
        return agent
    except FileNotFoundError:
        st.error("⚠️ Pre-trained model not found. Please run train.ipynb first.")
        return None


def main():
    st.set_page_config(
        page_title="Epsilon-Greedy Travel Recommender",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Epsilon-Greedy Travel Recommender")
    st.markdown("---")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        user_profiles, places, feature_columns = load_data()
        st.session_state.user_profiles = user_profiles
        st.session_state.places = places
        st.session_state.feature_columns = feature_columns
        st.session_state.agent = load_model()
        st.session_state.feedback_count = 0
    
    if st.session_state.agent is None:
        st.stop()
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        **Epsilon-Greedy** is a simple reinforcement learning algorithm that:
        - Explores random options with probability ε (epsilon)
        - Exploits the best known option with probability 1-ε
        - Doesn't consider user context
        - Learns which destinations are generally popular
        """)
        
        st.markdown("---")
        st.metric("Total Destinations", len(st.session_state.places))
        st.metric("Feedback Received", st.session_state.feedback_count)
    
    # Main content
    st.subheader("✈️ Top 10 Travel Recommendations")
    
    # Get top-10 recommendations
    ranked_indices = np.argsort(st.session_state.agent.values)[::-1][:10]
    
    # Display recommendations
    cols = st.columns(2)
    
    for i, idx in enumerate(ranked_indices):
        place = st.session_state.places[idx]
        col = cols[i % 2]
        
        with col:
            with st.container():
                st.markdown(f"### {i+1}. {place['name']}")
                st.write(f"**Type:** {place['type']}")
                st.write(f"**Rating:** {'⭐' * int(place['rating'])} ({place['rating']:.2f})")
                st.write(f"**Keywords:** {', '.join(place['keywords'][:3])}")
                st.write(f"**AI Score:** {st.session_state.agent.values[idx]:.3f}")
                
                # Feedback section
                col1, col2 = st.columns([3, 1])
                with col1:
                    rating = st.slider(
                        f"Rate this destination",
                        -1.0, 1.0, 0.0, 0.1,
                        key=f"rating_{idx}_{i}"
                    )
                with col2:
                    if st.button("Submit", key=f"submit_{idx}_{i}"):
                        st.session_state.agent.update(idx, rating)
                        st.session_state.feedback_count += 1
                        st.success("✅ Feedback recorded!")
                        st.rerun()
                
                st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Recommendations", use_container_width=True):
        st.rerun()


if __name__ == "__main__":
    main()
