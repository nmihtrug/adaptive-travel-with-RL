#!/bin/bash

# Quick start script for Adaptive Travel Recommendation Demo

echo "🌍 Adaptive Travel Recommendation System - Demo Launcher"
echo "=========================================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit is not installed."
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Check if models exist
if [ ! -f "../saved_models/epsilon_greedy.pkl" ]; then
    echo "⚠️  Pre-trained models not found!"
    echo "Please run notebooks/train.ipynb first to train the models."
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

echo ""
echo "Select a demo to run:"
echo "1. Main Demo (All algorithms)"
echo "2. Epsilon-Greedy Demo"
echo "3. LinUCB Demo"
echo "4. Thompson Sampling Demo"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Launching Main Demo..."
        streamlit run demo.py
        ;;
    2)
        echo "Launching Epsilon-Greedy Demo..."
        streamlit run interactive_recommendation_egreedy.py
        ;;
    3)
        echo "Launching LinUCB Demo..."
        streamlit run interactive_recommendation_linucb.py
        ;;
    4)
        echo "Launching Thompson Sampling Demo..."
        streamlit run interactive_recommendation_ts.py
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac
