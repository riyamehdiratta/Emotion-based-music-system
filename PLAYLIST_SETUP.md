# 🎵 Spotify Playlist Integration Guide

## Overview

You can now use your **custom Spotify playlist** instead of genre-based search! This gives you full control over which songs are recommended.

## Your Playlist

**Playlist ID**: `6K9AUU5oEIAwifcbJlDIiS`

This playlist will be used when playlist mode is enabled.

## Setup Instructions

### Step 1: Configure Environment Variables

Edit your `.env` file:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Enable playlist mode
USE_SPOTIFY_PLAYLIST=true

# Your playlist ID (already set to your playlist)
SPOTIFY_PLAYLIST_ID=6K9AUU5oEIAwifcbJlDIiS
```

### Step 2: Restart the Server

```bash
python app.py
```

You should see:
```
✅ Spotify client initialized successfully
🎵 Using Spotify playlist: 6K9AUU5oEIAwifcbJlDIiS
```

### Step 3: Test It!

1. Upload an image or use webcam
2. Detect emotion
3. Get recommendations from your playlist!

## How It Works

### Playlist Mode (USE_SPOTIFY_PLAYLIST=true)
- ✅ Fetches tracks from your playlist
- ✅ Shows top 5 tracks from playlist
- ✅ Displays full playlist embed at bottom
- ✅ All tracks are from your curated playlist

### Genre Mode (USE_SPOTIFY_PLAYLIST=false)
- ✅ Searches Spotify by genre
- ✅ Multilingual support
- ✅ Dynamic recommendations based on emotion

## Features

### Playlist Embed
When using playlist mode, you'll see:
- Individual track embeds (top 5)
- Full playlist embed at the bottom
- "Open in Spotify" buttons
- Album art for each track

### Playlist Embed Display
The full playlist embed shows:
- All tracks in your playlist
- Play/pause controls
- Track list
- Spotify branding

## Customizing Your Playlist

### Add Songs to Playlist
1. Go to your playlist on Spotify
2. Click "Add songs"
3. Search and add songs
4. The system will automatically use updated playlist

### Organize by Emotion
You can create multiple playlists:
- Happy playlist
- Sad playlist
- Angry playlist
- etc.

Then update `SPOTIFY_PLAYLIST_ID` in code or use different playlists per emotion.

## Code Integration

The playlist ID is used in:
- `recommender.py` - Fetches tracks from playlist
- `spotify_client.py` - `get_playlist_tracks()` method
- Frontend - Displays playlist embed

## Example: Multiple Playlists by Emotion

You can modify `recommender.py` to use different playlists:

```python
EMOTION_PLAYLIST_MAP = {
    'Happy': 'your_happy_playlist_id',
    'Sad': 'your_sad_playlist_id',
    'Angry': 'your_angry_playlist_id',
    # etc.
}
```

## Troubleshooting

### Playlist not showing
- Check `USE_SPOTIFY_PLAYLIST=true` in `.env`
- Verify playlist ID is correct
- Make sure playlist is public or you have access
- Restart server after changing `.env`

### No tracks found
- Check playlist has tracks
- Verify playlist ID is correct
- Check Spotify API credentials

### Playlist embed not loading
- Check internet connection
- Verify playlist is public
- Try opening playlist URL directly in browser

## Benefits of Playlist Mode

✅ **Full Control**: You choose exactly which songs appear  
✅ **Curated**: Pre-selected high-quality tracks  
✅ **Consistent**: Same songs every time  
✅ **Easy Updates**: Just update playlist on Spotify  
✅ **No API Limits**: Fewer API calls (fetch once)  

## Switching Between Modes

**Use Playlist**:
```env
USE_SPOTIFY_PLAYLIST=true
```

**Use Genre Search**:
```env
USE_SPOTIFY_PLAYLIST=false
```

## Your Current Setup

- **Playlist ID**: `6K9AUU5oEIAwifcbJlDIiS`
- **Mode**: Controlled by `USE_SPOTIFY_PLAYLIST` in `.env`
- **Default**: Genre search (playlist mode off)

---

**Ready to use!** Just set `USE_SPOTIFY_PLAYLIST=true` in your `.env` file! 🎵

