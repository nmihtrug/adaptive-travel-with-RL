"""Quick launcher for Adaptive Travel Recommendation demos."""

import os
import sys
import subprocess


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def check_models():
    """Check if pre-trained models exist."""
    model_dir = '../saved_models'
    models = ['epsilon_greedy.pkl', 'linucb.pkl', 'thompson_sampling.pkl']
    return all(os.path.exists(os.path.join(model_dir, model)) for model in models)


def main():
    print("🌍 Adaptive Travel Recommendation System - Demo Launcher")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("⚠️  Streamlit is not installed.")
        response = input("Install required packages? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        else:
            print("Cannot proceed without Streamlit. Exiting.")
            return
    
    # Check models
    if not check_models():
        print("⚠️  Pre-trained models not found!")
        print("Please run notebooks/train.ipynb first to train the models.")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print()
    print("Select a demo to run:")
    print("1. Main Demo (All algorithms)")
    print("2. Epsilon-Greedy Demo")
    print("3. LinUCB Demo")
    print("4. Thompson Sampling Demo")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    demos = {
        '1': 'demo.py',
        '2': 'interactive_recommendation_egreedy.py',
        '3': 'interactive_recommendation_linucb.py',
        '4': 'interactive_recommendation_ts.py'
    }
    
    if choice in demos:
        print(f"\nLaunching {demos[choice]}...")
        subprocess.run(['streamlit', 'run', demos[choice]])
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
