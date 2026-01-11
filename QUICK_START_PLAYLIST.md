# 🚀 Quick Start: Using Your Spotify Playlist

## Your Playlist
**Playlist ID**: `6K9AUU5oEIAwifcbJlDIiS`

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` File
```bash
cp .env.example .env
```

### 3. Edit `.env` File
Open `.env` and add your Spotify credentials:

```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here

# Enable playlist mode
USE_SPOTIFY_PLAYLIST=true

# Your playlist ID (already set)
SPOTIFY_PLAYLIST_ID=6K9AUU5oEIAwifcbJlDIiS
```

**Where to get credentials:**
1. Go to https://developer.spotify.com/dashboard
2. Log in with Spotify
3. Click "Create App"
4. Copy Client ID and Client Secret
5. Paste in `.env` file

### 4. Run the Server
```bash
python app.py
```

You should see:
```
✅ Spotify client initialized successfully
🎵 Using Spotify playlist: 6K9AUU5oEIAwifcbJlDIiS
```

### 5. Open in Browser
```
http://localhost:5001
```

### 6. Test It!
1. Upload an image with a face OR use webcam
2. Click "Detect Emotion"
3. See recommendations from your playlist!
4. Scroll down to see the full playlist embed

## What You'll See

✅ **Top 5 tracks** from your playlist  
✅ **Album art** for each track  
✅ **Individual track players**  
✅ **Full playlist embed** at the bottom  
✅ **"Open in Spotify"** buttons  

## Switching Modes

**Use Your Playlist:**
```env
USE_SPOTIFY_PLAYLIST=true
```

**Use Genre Search (default):**
```env
USE_SPOTIFY_PLAYLIST=false
```

## Troubleshooting

**"Spotify credentials not found"**
- Make sure `.env` file exists
- Check credentials are correct (no extra spaces)
- Restart server after editing `.env`

**"No tracks found"**
- Check playlist has songs
- Verify playlist ID is correct
- Make sure playlist is accessible

**Playlist embed not showing**
- Check internet connection
- Verify playlist is public
- Try opening playlist URL directly

## That's It! 🎵

Your playlist is now integrated! Just set `USE_SPOTIFY_PLAYLIST=true` and restart the server.

