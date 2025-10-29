#!/bin/bash

# Installation and Setup Script
# Run this to set up the adaptive travel recommendation system

echo "======================================================================"
echo "🚀 Adaptive Travel Recommendation System - Setup"
echo "======================================================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo ""
echo "======================================================================"
echo "🧪 Running Test Suite"
echo "======================================================================"
python test_setup.py

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Setup Complete!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Train models:     python main.py"
    echo "  2. Open notebooks:   jupyter notebook"
    echo "  3. Read docs:        cat README.md"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "⚠️  Some tests failed"
    echo "======================================================================"
    echo ""
    echo "This is likely due to missing dependencies."
    echo "Make sure all packages are installed:"
    echo "  pip install -r requirements.txt"
    echo ""
fi
