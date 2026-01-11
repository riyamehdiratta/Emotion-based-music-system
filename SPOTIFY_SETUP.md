# 🎵 Spotify Integration Setup Guide

## Overview

The Emotion-Based Music Recommendation System now uses **Spotify Web API** to fetch real, multilingual songs based on detected emotions.

## Prerequisites

1. A Spotify account (free or premium)
2. Python 3.8+
3. Internet connection

## Step 1: Create Spotify Developer App

1. **Go to Spotify Developer Dashboard**
   - Visit: https://developer.spotify.com/dashboard
   - Log in with your Spotify account

2. **Create a New App**
   - Click **"Create App"** button
   - Fill in the form:
     - **App Name**: `Emotion Music Recommender` (or any name)
     - **App Description**: `Music recommendation based on emotions`
     - **Website**: `http://localhost:5001` (or your domain)
     - **Redirect URI**: `http://localhost:5001/callback` (optional, not needed for Client Credentials)
   - Check the agreement checkbox
   - Click **"Save"**

3. **Get Your Credentials**
   - After creating the app, you'll see:
     - **Client ID**: Copy this
     - **Client Secret**: Click "View client secret" and copy it
   - ⚠️ **Keep these secret!** Don't share them publicly.

## Step 2: Configure Environment Variables

1. **Create `.env` file**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file**
   ```env
   SPOTIFY_CLIENT_ID=your_actual_client_id_here
   SPOTIFY_CLIENT_SECRET=your_actual_client_secret_here
   ```

3. **Replace the placeholders** with your actual credentials from Step 1.

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `requests` - For API calls
- `python-dotenv` - For loading `.env` file

## Step 4: Test the Integration

1. **Start the server**
   ```bash
   python app.py
   ```

2. **Check the console**
   - You should see: `✅ Spotify client initialized successfully`
   - If you see an error, check your credentials in `.env`

3. **Test emotion detection**
   - Upload an image or use webcam
   - The system should fetch real songs from Spotify

## How It Works

### Emotion → Genre Mapping

The system maps detected emotions to Spotify genres:

| Emotion | Genres |
|---------|--------|
| Happy | pop, dance, edm, happy, upbeat |
| Sad | acoustic, soul, blues, sad, melancholic |
| Angry | rock, metal, hard-rock, punk |
| Neutral | chill, lofi, ambient, indie |
| Surprise | indie, alternative, experimental |
| Fear | ambient, cinematic, dark-ambient, horror |
| Disgust | experimental, industrial, noise |

### Multilingual Support

The system searches across multiple markets:
- **US** - English songs
- **IN** - Hindi, Punjabi, Tamil, Telugu songs
- **ES** - Spanish songs (Spain)
- **MX** - Spanish songs (Mexico)
- **KR** - Korean songs

### Recommendation Logic

1. Detects emotion from face
2. Maps emotion to genres
3. Searches Spotify across multiple markets
4. Ranks by popularity and mood match
5. Ensures at least 3 different languages
6. Returns top 5 tracks

## Features

✅ **Real Songs**: Live data from Spotify  
✅ **Multilingual**: Songs in multiple languages  
✅ **Album Art**: High-quality cover images  
✅ **Spotify Embed**: Play songs directly in browser  
✅ **Open in Spotify**: Direct link to full song  
✅ **No Downloads**: Uses official Spotify API only  

## Troubleshooting

### Error: "Spotify credentials not found"

**Solution**: 
- Make sure `.env` file exists
- Check that `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` are set
- Restart the server after creating `.env`

### Error: "Invalid client credentials"

**Solution**:
- Double-check your Client ID and Secret
- Make sure there are no extra spaces
- Regenerate credentials in Spotify Dashboard if needed

### Error: "Rate limit exceeded"

**Solution**:
- Spotify API has rate limits
- Wait a few minutes and try again
- The system caches access tokens to minimize API calls

### No songs found for emotion

**Solution**:
- Some emotions/genres may have fewer results
- Try a different image with a clearer emotion
- The system will fallback to local database if Spotify fails

### Songs not playing

**Solution**:
- Spotify embeds require internet connection
- Some tracks may not have preview URLs
- Click "Open in Spotify" to play full song

## API Limits

- **Rate Limit**: 10,000 requests per hour (per app)
- **Token Expiry**: 1 hour (automatically refreshed)
- **Search Limit**: 50 results per query

## Security Notes

- ⚠️ **Never commit `.env` file to git**
- ⚠️ **Don't share your Client Secret publicly**
- ⚠️ **Use environment variables in production**
- ✅ `.env` is already in `.gitignore`

## Editing Emotion-Genre Mapping

To customize the emotion-to-genre mapping, edit `recommender.py`:

```python
EMOTION_GENRE_MAP = {
    'Happy': ['pop', 'dance', 'edm', 'happy', 'upbeat'],
    'Sad': ['acoustic', 'soul', 'blues', 'sad', 'melancholic'],
    # Add or modify as needed
}
```

## Support

For Spotify API issues:
- Spotify Developer Documentation: https://developer.spotify.com/documentation/web-api
- Spotify Community: https://community.spotify.com/t5/Spotify-for-Developers/bd-p/Spotify_Developer

---

**Status**: ✅ Ready to use with Spotify integration!

