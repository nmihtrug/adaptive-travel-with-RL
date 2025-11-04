"""Main Streamlit Demo - Adaptive Travel Recommendation System."""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('..')

from models.agents import EpsilonGreedyAgent, LinUCBAgent, ContextualThompsonSampling


@st.cache_data
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


def load_models():
    """Load all pre-trained models."""
    models = {}
    try:
        with open('../saved_models/epsilon_greedy.pkl', 'rb') as f:
            models['Epsilon-Greedy'] = pickle.load(f)
        with open('../saved_models/linucb.pkl', 'rb') as f:
            models['LinUCB'] = pickle.load(f)
        with open('../saved_models/thompson_sampling.pkl', 'rb') as f:
            models['Thompson Sampling'] = pickle.load(f)
        return models
    except FileNotFoundError as e:
        st.error(f"⚠️ Pre-trained model not found: {e}. Please run train.ipynb first.")
        return None


def context_to_vector(user, feature_columns):
    """Convert user preferences to vector."""
    return np.array([user["prefs"].get(col, 0) for col in feature_columns])


def get_recommendations_egreedy(agent, places, top_k=10):
    """Get top-k recommendations from Epsilon-Greedy."""
    ranked_indices = np.argsort(agent.values)[::-1][:top_k]
    recommendations = []
    for idx in ranked_indices:
        recommendations.append({
            'idx': idx,
            'name': places[idx]['name'],
            'type': places[idx]['type'],
            'rating': places[idx]['rating'],
            'keywords': places[idx]['keywords'],
            'score': agent.values[idx]
        })
    return recommendations


def get_recommendations_linucb(agent, places, user, feature_columns, top_k=10):
    """Get top-k recommendations from LinUCB."""
    x = context_to_vector(user, feature_columns)
    scores = []
    for arm in range(len(places)):
        A_inv = np.linalg.inv(agent.A[arm])
        theta = A_inv @ agent.b[arm]
        score = theta @ x + agent.alpha * np.sqrt(x @ A_inv @ x)
        scores.append(score)
    
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = []
    for idx in ranked_indices:
        recommendations.append({
            'idx': idx,
            'name': places[idx]['name'],
            'type': places[idx]['type'],
            'rating': places[idx]['rating'],
            'keywords': places[idx]['keywords'],
            'score': scores[idx],
            'user_pref': user['prefs'].get(places[idx]['type'], 0)
        })
    return recommendations, x


def get_recommendations_ts(agent, places, user, feature_columns, top_k=10):
    """Get top-k recommendations from Thompson Sampling."""
    x = context_to_vector(user, feature_columns)
    sampled_rewards = []
    for arm in range(agent.n_arms):
        B_inv = np.linalg.inv(agent.B[arm])
        mu_hat = B_inv @ agent.f[arm]
        theta_sample = np.random.multivariate_normal(mu_hat, agent.alpha**2 * B_inv)
        sampled_rewards.append(theta_sample @ x)
    
    ranked_indices = np.argsort(sampled_rewards)[::-1][:top_k]
    recommendations = []
    for idx in ranked_indices:
        recommendations.append({
            'idx': idx,
            'name': places[idx]['name'],
            'type': places[idx]['type'],
            'rating': places[idx]['rating'],
            'keywords': places[idx]['keywords'],
            'score': sampled_rewards[idx],
            'user_pref': user['prefs'].get(places[idx]['type'], 0)
        })
    return recommendations, x


def main():
    st.set_page_config(
        page_title="Adaptive Travel Recommender",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🌍 Adaptive Travel Recommendation System")
    st.markdown("### Powered by Reinforcement Learning")
    st.markdown("---")
    
    # Load data
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading data and models..."):
            user_profiles, places, feature_columns = load_data()
            models = load_models()
            
            if models is None:
                st.stop()
            
            st.session_state.user_profiles = user_profiles
            st.session_state.places = places
            st.session_state.feature_columns = feature_columns
            st.session_state.models = models
            st.session_state.data_loaded = True
            st.session_state.feedback_count = {name: 0 for name in models.keys()}
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Algorithm",
            list(st.session_state.models.keys()),
            help="Choose which RL algorithm to use for recommendations"
        )
        
        st.markdown("---")
        
        # User selection for contextual models
        if selected_model != 'Epsilon-Greedy':
            st.subheader("👤 User Profile")
            selected_user_id = st.selectbox(
                "Select User",
                range(len(st.session_state.user_profiles)),
                format_func=lambda x: f"User {x}"
            )
            
            user = st.session_state.user_profiles[selected_user_id]
            
            # Show top preferences
            st.write("**Top 5 Preferences:**")
            sorted_prefs = sorted(user['prefs'].items(), key=lambda x: x[1], reverse=True)[:5]
            for pref, score in sorted_prefs:
                st.write(f"• {pref}: {score:.1%}")
        else:
            selected_user_id = None
            user = None
        
        st.markdown("---")
        
        # Number of recommendations
        top_k = st.slider("Number of Recommendations", 5, 20, 10)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Total Destinations", len(st.session_state.places))
        st.metric("Total Users", len(st.session_state.user_profiles))
        st.metric("Feedback Count", st.session_state.feedback_count[selected_model])
        
        st.markdown("---")
        
        # Model info
        st.subheader("ℹ️ About Algorithm")
        if selected_model == 'Epsilon-Greedy':
            st.info("""
            **Epsilon-Greedy**
            - Simple exploration-exploitation
            - No user context
            - Learns popular destinations
            - Fast and efficient
            """)
        elif selected_model == 'LinUCB':
            st.info("""
            **LinUCB**
            - Contextual bandit
            - Personalized recommendations
            - Confidence bounds
            - Balances exploration
            """)
        else:
            st.info("""
            **Thompson Sampling**
            - Bayesian approach
            - Probability matching
            - Natural exploration
            - Contextual learning
            """)
    
    # Main content area
    st.subheader(f"✈️ Top {top_k} Travel Recommendations")
    if selected_model != 'Epsilon-Greedy':
        st.caption(f"Personalized for User {selected_user_id}")
    
    # Get recommendations
    agent = st.session_state.models[selected_model]
    
    if selected_model == 'Epsilon-Greedy':
        recommendations = get_recommendations_egreedy(agent, st.session_state.places, top_k)
        context_vector = None
    elif selected_model == 'LinUCB':
        recommendations, context_vector = get_recommendations_linucb(
            agent, st.session_state.places, user, st.session_state.feature_columns, top_k
        )
    else:  # Thompson Sampling
        recommendations, context_vector = get_recommendations_ts(
            agent, st.session_state.places, user, st.session_state.feature_columns, top_k
        )
    
    # Display recommendations in cards
    for i in range(0, len(recommendations), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            if i + j < len(recommendations):
                rec = recommendations[i + j]
                
                with col:
                    with st.container():
                        # Card header
                        st.markdown(f"### {i+j+1}. {rec['name']}")
                        
                        # Details
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Type:** {rec['type']}")
                            st.write(f"**Keywords:** {', '.join(rec['keywords'][:3])}")
                        with col2:
                            st.write(f"**Rating:** {rec['rating']:.2f}")
                            st.write(f"**AI Score:** {rec['score']:.3f}")
                        
                        # User preference match (for contextual models)
                        if 'user_pref' in rec:
                            st.progress(rec['user_pref'], text=f"Match: {rec['user_pref']:.1%}")
                        
                        # Feedback section
                        with st.expander("💬 Provide Feedback"):
                            rating = st.slider(
                                "How interested are you?",
                                0.0, 1.0, 0.5, 0.1,
                                key=f"rating_{selected_model}_{rec['idx']}_{i}_{j}"
                            )
                            
                            if st.button("Submit Feedback", key=f"submit_{selected_model}_{rec['idx']}_{i}_{j}"):
                                # Update model
                                if selected_model == 'Epsilon-Greedy':
                                    agent.update(rec['idx'], rating)
                                else:
                                    agent.update(rec['idx'], context_vector, rating)
                                
                                st.session_state.feedback_count[selected_model] += 1
                                st.success("✅ Thank you for your feedback!")
                                st.rerun()
                        
                        st.markdown("---")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Recommendations", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("💾 Save Current Model", use_container_width=True):
            try:
                model_map = {
                    'Epsilon-Greedy': 'epsilon_greedy',
                    'LinUCB': 'linucb',
                    'Thompson Sampling': 'thompson_sampling'
                }
                filename = f"../saved_models/{model_map[selected_model]}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(agent, f)
                st.success(f"✅ {selected_model} model saved!")
            except Exception as e:
                st.error(f"Error saving model: {e}")
    
    with col3:
        if st.button("🔄 Reset Feedback Count", use_container_width=True):
            st.session_state.feedback_count[selected_model] = 0
            st.rerun()


if __name__ == "__main__":
    main()
