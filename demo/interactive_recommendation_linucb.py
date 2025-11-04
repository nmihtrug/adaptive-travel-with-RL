"""Interactive LinUCB Recommendation Demo."""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('..')

from models.agents import LinUCBAgent


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


def context_to_vector(user, feature_columns):
    """Convert user preferences to vector."""
    return np.array([user["prefs"].get(col, 0) for col in feature_columns])


def load_model():
    """Load pre-trained LinUCB model."""
    try:
        with open('../saved_models/linucb.pkl', 'rb') as f:
            agent = pickle.load(f)
        return agent
    except FileNotFoundError:
        st.error("⚠️ Pre-trained model not found. Please run train.ipynb first.")
        return None


def main():
    st.set_page_config(
        page_title="LinUCB Travel Recommender",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 LinUCB Contextual Travel Recommender")
    st.markdown("---")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        user_profiles, places, feature_columns = load_data()
        st.session_state.user_profiles = user_profiles
        st.session_state.places = places
        st.session_state.feature_columns = feature_columns
        st.session_state.agent = load_model()
        st.session_state.feedback_count = 0
        st.session_state.selected_user = 0
    
    if st.session_state.agent is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        **LinUCB** is a contextual bandit algorithm that:
        - Considers user preferences (context)
        - Uses confidence bounds for exploration
        - Provides personalized recommendations
        - Balances exploration vs exploitation
        """)
        
        st.markdown("---")
        
        # User selection
        st.subheader("👤 Select User Profile")
        st.session_state.selected_user = st.selectbox(
            "User ID",
            range(len(st.session_state.user_profiles)),
            format_func=lambda x: f"User {x}"
        )
        
        # Show user preferences
        user = st.session_state.user_profiles[st.session_state.selected_user]
        st.write("**Top Preferences:**")
        sorted_prefs = sorted(user['prefs'].items(), key=lambda x: x[1], reverse=True)[:5]
        for pref, score in sorted_prefs:
            st.write(f"- {pref}: {score:.3f}")
        
        st.markdown("---")
        st.metric("Total Destinations", len(st.session_state.places))
        st.metric("Feedback Received", st.session_state.feedback_count)
    
    # Main content
    user = st.session_state.user_profiles[st.session_state.selected_user]
    x = context_to_vector(user, st.session_state.feature_columns)
    
    st.subheader(f"✈️ Top 10 Personalized Recommendations for User {st.session_state.selected_user}")
    
    # Calculate scores for all places
    scores = []
    for arm in range(len(st.session_state.places)):
        A_inv = np.linalg.inv(st.session_state.agent.A[arm])
        theta = A_inv @ st.session_state.agent.b[arm]
        score = theta @ x + st.session_state.agent.alpha * np.sqrt(x @ A_inv @ x)
        scores.append(score)
    
    # Get top-10 recommendations
    ranked_indices = np.argsort(scores)[::-1][:10]
    
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
                st.write(f"**LinUCB Score:** {scores[idx]:.3f}")
                
                # User preference match
                pref_match = user['prefs'].get(place['type'], 0)
                st.progress(pref_match, text=f"Your preference: {pref_match:.2%}")
                
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
                        st.session_state.agent.update(idx, x, rating)
                        st.session_state.feedback_count += 1
                        st.success("✅ Feedback recorded!")
                        st.rerun()
                
                st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Recommendations", use_container_width=True):
        st.rerun()


if __name__ == "__main__":
    main()
