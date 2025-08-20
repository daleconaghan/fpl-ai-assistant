#!/bin/bash

# GitHub setup script for FPL AI Assistant
# Replace 'YOUR_USERNAME' with your actual GitHub username

echo "🚀 Setting up GitHub repository for FPL AI Assistant"
echo "=================================================="

# Set your GitHub username here
read -p "Enter your GitHub username: " USERNAME

if [ -z "$USERNAME" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

echo "📡 Adding GitHub remote origin..."
git remote add origin https://github.com/$USERNAME/fpl-ai-assistant.git

echo "🔍 Checking git status..."
git status

echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "✅ Repository successfully pushed to GitHub!"
echo "🌐 Your repository is now available at:"
echo "   https://github.com/$USERNAME/fpl-ai-assistant"

echo ""
echo "🎯 Next steps:"
echo "1. Visit your repository on GitHub"
echo "2. Add repository topics: machine-learning, fantasy-football, premier-league, python, streamlit"
echo "3. Consider adding a license (MIT recommended)"
echo "4. Star your own repository! ⭐"