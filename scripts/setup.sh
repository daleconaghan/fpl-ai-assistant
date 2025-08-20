#!/bin/bash

# FPL AI Assistant Setup Script

echo "🏆 Setting up FPL AI Assistant..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/raw/fpl
mkdir -p data/processed
mkdir -p data/predictions
mkdir -p logs
mkdir -p models/saved

# Test FPL API connection
echo "🧪 Testing FPL API connection..."
python src/data/fpl_api.py

echo "✅ FPL AI Assistant setup complete!"
echo ""
echo "🚀 To get started:"
echo "1. source venv/bin/activate"
echo "2. python scripts/collect_fpl_data.py"
echo "3. streamlit run app.py"
echo ""
echo "📊 Your FPL AI Assistant is ready to dominate your mini-league!"