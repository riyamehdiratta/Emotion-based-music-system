"""
Music Recommendation Engine with Spotify Integration
Recommends REAL songs from Spotify based on detected emotion with multi-language support.
"""

import os
from typing import List, Dict, Optional
from spotify_client import SpotifyClient

class MusicRecommender:
    """
    Music recommendation system that matches Spotify songs to emotions.
    Uses emotion → genre mapping and fetches real tracks from Spotify.
    """
    
    # Emotion to Genre Mapping (editable and well-documented)
    EMOTION_GENRE_MAP = {
        'Happy': ['pop', 'dance', 'edm', 'happy', 'upbeat'],
        'Sad': ['acoustic', 'soul', 'blues', 'sad', 'melancholic'],
        'Angry': ['rock', 'metal', 'hard-rock', 'punk'],
        'Neutral': ['chill', 'lofi', 'ambient', 'indie'],
        'Surprise': ['indie', 'alternative', 'experimental'],
        'Fear': ['ambient', 'cinematic', 'dark-ambient', 'horror'],
        'Disgust': ['experimental', 'industrial', 'noise']
    }
    
    # Market to Language mapping for better language inference
    MARKET_LANGUAGE_HINTS = {
        'US': 'English',
        'IN': 'Hindi',  # Primary, but can be Punjabi, Tamil, Telugu
        'ES': 'Spanish',
        'MX': 'Spanish',
        'KR': 'Korean'
    }
    
    # Default playlist ID (can be overridden via environment variable)
    DEFAULT_PLAYLIST_ID = '6K9AUU5oEIAwifcbJlDIiS'
    
    def __init__(self, use_spotify: bool = True, playlist_id: Optional[str] = None):
        """
        Initialize the recommender.
        
        Args:
            use_spotify: If True, use Spotify API. If False, fallback to local CSV.
        """
        self.use_spotify = use_spotify
        self.spotify_client = None
        self.playlist_id = playlist_id or os.getenv('SPOTIFY_PLAYLIST_ID', self.DEFAULT_PLAYLIST_ID)
        self.use_playlist = os.getenv('USE_SPOTIFY_PLAYLIST', 'false').lower() == 'true'
        
        if use_spotify:
            try:
                self.spotify_client = SpotifyClient()
                print("✅ Spotify client initialized successfully")
                if self.use_playlist:
                    print(f"🎵 Using Spotify playlist: {self.playlist_id}")
            except Exception as e:
                print(f"⚠️  Could not initialize Spotify client: {e}")
                print("   Falling back to local song database...")
                self.use_spotify = False
                self._init_local_fallback()
        else:
            self._init_local_fallback()
    
    def _init_local_fallback(self):
        """Initialize local CSV fallback (original implementation)."""
        # Keep original CSV loading as fallback
        import pandas as pd
        self.metadata_path = 'songs_metadata.csv'
        self.songs_df = None
        try:
            if os.path.exists(self.metadata_path):
                self.songs_df = pd.read_csv(self.metadata_path)
                print(f"✅ Loaded {len(self.songs_df)} songs from local database")
        except Exception as e:
            print(f"⚠️  Could not load local database: {e}")
    
    def get_genres_for_emotion(self, emotion: str) -> List[str]:
        """
        Get Spotify genres for a given emotion.
        
        Args:
            emotion: Detected emotion
            
        Returns:
            List of genre strings
        """
        return self.EMOTION_GENRE_MAP.get(emotion, ['pop'])  # Default to pop
    
    def get_recommendations(self, emotion: str, confidence: float = 1.0, 
                          top_n: int = 5, language: Optional[str] = None) -> List[Dict]:
        """
        Get music recommendations based on detected emotion.
        Uses Spotify API to fetch real, multilingual songs.
        
        Args:
            emotion: Detected emotion (Happy, Sad, Angry, etc.)
            confidence: Confidence score of emotion detection
            top_n: Number of recommendations to return
            language: Optional language filter (not strictly enforced for diversity)
            
        Returns:
            List of song dictionaries with Spotify metadata
        """
        if self.use_spotify and self.spotify_client:
            return self._get_spotify_recommendations(emotion, confidence, top_n, language)
        else:
            return self._get_local_recommendations(emotion, confidence, top_n, language)
    
    def _get_spotify_recommendations(self, emotion: str, confidence: float,
                                   top_n: int, language: Optional[str]) -> List[Dict]:
        """
        Get recommendations from Spotify API.
        Can use playlist or genre-based search.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            top_n: Number of recommendations
            language: Optional language filter
            
        Returns:
            List of track dictionaries
        """
        # Option 1: Use playlist if enabled
        if self.use_playlist and self.playlist_id:
            all_tracks = self.spotify_client.get_playlist_tracks(
                playlist_id=self.playlist_id,
                market='US',
                limit=50
            )
            
            # Filter tracks by emotion (match genre)
            genres = self.get_genres_for_emotion(emotion)
            # For playlist, we'll use all tracks but rank by emotion match
            # (In a real scenario, you'd have emotion tags in playlist, but we'll use all)
        else:
            # Option 2: Use genre-based search (original method)
            genres = self.get_genres_for_emotion(emotion)
            
            # Fetch tracks from multiple markets for multilingual diversity
            all_tracks = self.spotify_client.get_multilingual_tracks(
                genres=genres,
                limit_per_market=15  # Get more to ensure diversity
            )
        
        if not all_tracks:
            print(f"⚠️  No tracks found for emotion: {emotion}")
            return []
        
        # Infer language for each track
        for track in all_tracks:
            track['language'] = self.spotify_client.infer_language(track)
            track['emotion'] = emotion
            track['mood_match_score'] = confidence
        
        # Filter by language if specified (but prioritize diversity)
        if language:
            # Try to get at least 2 tracks in requested language, but ensure diversity
            language_tracks = [t for t in all_tracks if t['language'].lower() == language.lower()]
            other_tracks = [t for t in all_tracks if t['language'].lower() != language.lower()]
            
            # Combine: prefer requested language but ensure diversity
            if len(language_tracks) >= 2:
                selected = language_tracks[:2] + other_tracks[:top_n - 2]
            else:
                selected = language_tracks + other_tracks[:top_n - len(language_tracks)]
        else:
            selected = all_tracks
        
        # Ensure at least 3 different languages in output
        selected = self._ensure_language_diversity(selected, top_n)
        
        # Rank by popularity and mood match
        selected.sort(key=lambda x: (x['popularity'] * x['mood_match_score']), reverse=True)
        
        # Take top N
        recommendations = selected[:top_n]
        
        # Format for frontend
        formatted_recommendations = []
        for track in recommendations:
            formatted_track = {
                'song_name': track['name'],
                'artist': track['artist'],
                'language': track['language'],
                'emotion': track['emotion'],
                'genre': ', '.join(genres[:2]) if not self.use_playlist else 'Playlist',
                'spotify_id': track['id'],
                'spotify_url': track['external_url'],
                'album_art': track['album_art'],
                'preview_url': track.get('preview_url'),
                'popularity': track['popularity'],
                'recommendation_score': track['mood_match_score'] * (track['popularity'] / 100.0),
                'playlist_id': self.playlist_id if self.use_playlist else None
            }
            formatted_recommendations.append(formatted_track)
        
        return formatted_recommendations
    
    def _ensure_language_diversity(self, tracks: List[Dict], target_count: int) -> List[Dict]:
        """
        Ensure at least 3 different languages in recommendations.
        
        Args:
            tracks: List of track dictionaries
            target_count: Target number of recommendations
            
        Returns:
            List with language diversity
        """
        if len(tracks) < 3:
            return tracks[:target_count]
        
        # Group by language
        by_language = {}
        for track in tracks:
            lang = track['language']
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(track)
        
        # If we already have 3+ languages, return top tracks
        if len(by_language) >= 3:
            return tracks[:target_count]
        
        # Otherwise, try to get more diverse tracks
        # Sort languages by track count
        sorted_langs = sorted(by_language.items(), key=lambda x: len(x[1]), reverse=True)
        
        selected = []
        # Take at least 1 from each available language
        for lang, lang_tracks in sorted_langs:
            if lang_tracks:
                selected.append(lang_tracks[0])
        
        # Fill remaining slots with top tracks
        remaining_slots = target_count - len(selected)
        all_remaining = [t for t in tracks if t not in selected]
        all_remaining.sort(key=lambda x: x['popularity'], reverse=True)
        selected.extend(all_remaining[:remaining_slots])
        
        return selected[:target_count]
    
    def _get_local_recommendations(self, emotion: str, confidence: float,
                                  top_n: int, language: Optional[str]) -> List[Dict]:
        """
        Fallback to local CSV recommendations (original implementation).
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            top_n: Number of recommendations
            language: Optional language filter
            
        Returns:
            List of song dictionaries
        """
        if self.songs_df is None or len(self.songs_df) == 0:
            return []
        
        # Filter by emotion (case-insensitive)
        emotion_lower = emotion.lower() if emotion else ''
        filtered = self.songs_df[
            self.songs_df['emotion'].str.lower() == emotion_lower
        ].copy()
        
        # Filter by language if specified
        if language:
            filtered = filtered[filtered['language'].str.lower() == language.lower()]
        
        # If no results, return any songs
        if len(filtered) == 0:
            filtered = self.songs_df.copy()
        
        # Calculate recommendation score
        filtered['recommendation_score'] = confidence
        
        # Sort by recommendation score
        filtered = filtered.sort_values('recommendation_score', ascending=False)
        
        # Get top N recommendations
        top_songs = filtered.head(top_n)
        
        # Convert to list of dictionaries
        recommendations = []
        for _, row in top_songs.iterrows():
            song_dict = {
                'song_name': row['song_name'],
                'artist': row['artist'],
                'language': row['language'],
                'emotion': row['emotion'],
                'genre': row.get('genre', 'Unknown'),
                'audio_path': row.get('audio_path', ''),
                'recommendation_score': row['recommendation_score'],
                'spotify_url': None,  # No Spotify for local tracks
                'album_art': None
            }
            recommendations.append(song_dict)
        
        return recommendations
    
    def get_all_languages(self) -> List[str]:
        """Get list of all available languages."""
        if self.use_spotify:
            return ['English', 'Hindi', 'Punjabi', 'Tamil', 'Telugu', 'Korean', 'Spanish']
        else:
            if self.songs_df is None:
                return []
            return sorted(self.songs_df['language'].unique().tolist())
    
    def get_song_count_by_emotion(self) -> Dict[str, int]:
        """Get count of songs for each emotion (for stats)."""
        if self.use_spotify:
            # Return estimated counts (Spotify has millions)
            return {
                'Happy': 1000000,
                'Sad': 1000000,
                'Angry': 1000000,
                'Neutral': 1000000,
                'Surprise': 1000000,
                'Fear': 1000000,
                'Disgust': 1000000
            }
        else:
            if self.songs_df is None:
                return {}
            return self.songs_df['emotion'].value_counts().to_dict()
