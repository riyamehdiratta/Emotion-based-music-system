#!/bin/bash

# Setup script for Emotion-Based Music Recommendation System

echo "🎵 Setting up Emotion-Based Music Recommendation System..."
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p static/songs/english
mkdir -p static/songs/hindi
mkdir -p static/songs/punjabi
mkdir -p static/songs/tamil
mkdir -p static/songs/telugu
mkdir -p static/songs/korean
mkdir -p static/songs/spanish
mkdir -p models
mkdir -p uploads
mkdir -p templates
mkdir -p static/css
mkdir -p static/js

echo "✅ Directories created!"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
    echo ""
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: python app.py"
echo "  3. Open browser: http://localhost:5000"
echo ""
echo "Note: Add your audio files to static/songs/[language]/ directories"
echo "      The metadata CSV will be auto-generated on first run."

