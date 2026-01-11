#!/bin/bash

echo "🎵 Setting up Emotion Music Recommender with Spotify Playlist..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your Spotify credentials:"
    echo "   SPOTIFY_CLIENT_ID=your_id"
    echo "   SPOTIFY_CLIENT_SECRET=your_secret"
    echo "   USE_SPOTIFY_PLAYLIST=true"
    echo ""
    read -p "Press Enter after editing .env file..."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting server..."
echo "Access at: http://localhost:5001"
echo ""

python app.py
