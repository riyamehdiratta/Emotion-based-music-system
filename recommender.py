"""
Music Recommendation Engine with Spotify Integration
Recommends REAL songs from Spotify based on detected emotion with multi-language support.
"""

import os
from typing import List, Dict, Optional
from spotify_client import SpotifyClient
from youtube_client import YouTubeClient
import pandas as pd


# Required dataset columns
REQUIRED_COLUMNS = {'song_name', 'artist', 'language', 'emotion'}
URL_COLUMNS = {'youtube_url', 'spotify_url'}

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
        
        # Initialize Spotify if requested
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

        # Initialize YouTube client for strict language recommendations
        try:
            self.youtube_client = YouTubeClient()
            print("✅ YouTube client initialized")
        except Exception as e:
            self.youtube_client = None
            print(f"⚠️ YouTube client not initialized: {e}")
        # Always attempt to load local dataset once (used as fallback and for strict language lists)
        self.songs_df = None
        self.metadata_path = 'songs_metadata.csv'
        self._init_local_fallback()
    
    def _init_local_fallback(self):
        """Initialize local CSV fallback (original implementation)."""
        # Load CSV or JSON dataset once and validate columns
        try:
            if os.path.exists(self.metadata_path):
                df = pd.read_csv(self.metadata_path)
            else:
                # try json
                json_path = self.metadata_path.replace('.csv', '.json')
                if os.path.exists(json_path):
                    df = pd.read_json(json_path)
                else:
                    df = pd.DataFrame()

            # Validate columns
            if not df.empty:
                cols = set(df.columns.str.lower())
                missing = REQUIRED_COLUMNS - set([c.lower() for c in df.columns])
                if missing:
                    print(f"⚠️  Dataset missing required columns: {missing}")
                    self.songs_df = pd.DataFrame(columns=list(REQUIRED_COLUMNS) + list(URL_COLUMNS))
                else:
                    # Normalize column names to expected casing
                    df.columns = [c.lower() for c in df.columns]
                    # Ensure url columns exist
                    for u in URL_COLUMNS:
                        if u not in df.columns:
                            df[u] = None
                    # Standardize language and emotion capitalization
                    df['language'] = df['language'].astype(str).str.strip()
                    df['emotion'] = df['emotion'].astype(str).str.strip()
                    self.songs_df = df.rename(columns={
                        'song_name': 'song_name',
                        'artist': 'artist',
                        'language': 'language',
                        'emotion': 'emotion'
                    })
                    print(f"✅ Loaded {len(self.songs_df)} songs from local database")
            else:
                print("⚠️  No local dataset found; songs_df initialized empty")
                self.songs_df = pd.DataFrame(columns=list(REQUIRED_COLUMNS) + list(URL_COLUMNS))
        except Exception as e:
            print(f"⚠️  Could not load local database: {e}")
            self.songs_df = pd.DataFrame(columns=list(REQUIRED_COLUMNS) + list(URL_COLUMNS))
    
    def get_genres_for_emotion(self, emotion: str) -> List[str]:
        """
        Get Spotify genres for a given emotion.
        
        Args:
            emotion: Detected emotion
            
        Returns:
            List of genre strings
        """
        if not emotion:
            return ['pop']

        key = emotion.strip().capitalize()
        # Special handling: Neutral/Uncertain should map to uplifting genres
        # Neutral does not mean sad — recommending uplifting content improves engagement
        if key in ('Neutral', 'Uncertain'):
            # Combine Happy + Neutral (chill/upbeat) genres and dedupe
            happy = self.EMOTION_GENRE_MAP.get('Happy', [])
            neutral = self.EMOTION_GENRE_MAP.get('Neutral', [])
            combined = list(dict.fromkeys(happy + neutral))
            return combined

        return self.EMOTION_GENRE_MAP.get(key, ['pop'])  # Default to pop
    
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
        # Prefer YouTube deterministic recommendations if client available
        if self.youtube_client:
            return self._get_youtube_recommendations(emotion, confidence, top_n, language)
        if self.use_spotify and self.spotify_client:
            return self._get_spotify_recommendations(emotion, confidence, top_n, language)
        return self._get_local_recommendations(emotion, confidence, top_n, language)

    def _language_keyword(self, language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        # Standardize language keywords
        mapping = {
            'english': 'English',
            'hindi': 'Hindi',
            'punjabi': 'Punjabi',
            'spanish': 'Spanish'
        }
        key = language.strip().lower()
        return mapping.get(key, language)

    def _get_youtube_recommendations(self, emotion: str, confidence: float, top_n: int, language: Optional[str]) -> List[Dict]:
        """
        Strict language-based YouTube recommendations.
        Query format: emotion + mood + "song" + selected_language
        Strict filter: title OR channel name must contain the language keyword.
        Ranking: emotion-to-mood match, search relevance (order), view count.
        """
        if not self.youtube_client:
            return []

        # Build mood words from genres map
        moods = self.get_genres_for_emotion(emotion)
        mood_phrase = moods[0] if moods else ''

        lang_keyword = self._language_keyword(language)
        # Build deterministic query
        q_parts = [emotion.lower(), mood_phrase.lower(), 'song']
        if lang_keyword:
            q_parts.append(lang_keyword.lower())
        query = ' '.join([p for p in q_parts if p])

        videos = self.youtube_client.search_videos(query, max_results=50)
        if not videos:
            return []

        # Strict filtering: title or channel must contain language keyword
        if lang_keyword:
            filtered = []
            for v in videos:
                title = (v.get('title') or '').lower()
                channel = (v.get('channel_title') or '').lower()
                if lang_keyword.lower() in title or lang_keyword.lower() in channel:
                    filtered.append(v)
        else:
            filtered = videos

        if not filtered:
            return []

        # Scoring
        # Mood match: presence of mood keywords in title/description
        scored = []
        max_views = max([v['viewCount'] for v in filtered]) if filtered else 1
        for rank, v in enumerate(filtered):
            title = (v.get('title') or '').lower()
            desc = (v.get('description') or '').lower()
            mood_score = 0.0
            for m in moods:
                if m.lower() in title or m.lower() in desc:
                    mood_score = 1.0
                    break

            # Relevance score approximated by inverse rank
            relevance = 1.0 - (rank / max(1, len(filtered) - 1)) if len(filtered) > 1 else 1.0
            view_norm = (v['viewCount'] / max_views) if max_views > 0 else 0.0

            recommendation_score = 0.5 * mood_score + 0.3 * relevance + 0.2 * view_norm
            scored.append((recommendation_score, v))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [v for _, v in scored[:top_n]]

        results = []
        for v in top:
            results.append({
                'song_name': v.get('title'),
                'artist': v.get('channel_title'),
                'language': lang_keyword or 'Unknown',
                'emotion': emotion,
                'genre': mood_phrase,
                'youtube_id': v.get('id'),
                'youtube_url': v.get('url'),
                'thumbnail': (v.get('thumbnails') or {}).get('high', {}).get('url') if v.get('thumbnails') else None,
                'view_count': v.get('viewCount'),
                'recommendation_score': None
            })

        return results
    
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
        # Ensure songs_df is present and non-empty
        if self.songs_df is None or len(self.songs_df) == 0:
            return []

        # Default language to English if not provided
        if not language:
            language = 'English'

        # 1) Filter by emotion (case-insensitive)
        emotion_lower = (emotion or '').strip().lower()

        # Special-case: Neutral or Uncertain -> treat as request for uplifting content
        if emotion_lower in ('neutral', 'uncertain'):
            # Map to happy + chill genres; prefer songs labeled 'Happy' OR songs whose genre contains uplifting keywords
            uplifting_emotions = ['happy']
            uplift_keywords = ['chill', 'lofi', 'upbeat', 'pop', 'dance', 'indie']

            # Songs explicitly labeled as 'Happy'
            df_emotion = self.songs_df[self.songs_df['emotion'].str.strip().str.lower().isin(uplifting_emotions)].copy()
            # If genre column exists, include songs whose genre matches uplifting keywords
            if 'genre' in self.songs_df.columns:
                mask = self.songs_df['genre'].astype(str).str.lower().apply(lambda g: any(k in g for k in uplift_keywords))
                df_genre_uplift = self.songs_df[mask].copy()
                # union
                df_emotion = pd.concat([df_emotion, df_genre_uplift]).drop_duplicates()
        else:
            df_emotion = self.songs_df[self.songs_df['emotion'].str.strip().str.lower() == emotion_lower].copy()

        # If no songs for this emotion, return empty (caller will handle fallback message)
        if df_emotion.empty:
            return []

        # 2) Filter by selected language (strict, case-insensitive)
        lang_lower = language.strip().lower()
        df_lang = df_emotion[df_emotion['language'].str.strip().str.lower() == lang_lower].copy()

        # If no songs exist for emotion+language, return empty to allow fallback handling
        if df_lang.empty:
            return []

        # 3) Sort by relevance_score (if present), then popularity/viewCount, desc
        sort_keys = []
        if 'relevance_score' in df_lang.columns:
            sort_keys.append('relevance_score')
        # common popularity columns
        pop_col = None
        for c in ['popularity', 'viewcount', 'view_count', 'views']:
            if c in df_lang.columns:
                pop_col = c
                break
        if pop_col:
            sort_keys.append(pop_col)

        if sort_keys:
            df_lang = df_lang.sort_values(by=sort_keys, ascending=False)
        else:
            # fallback: alphabetical by song_name
            df_lang = df_lang.sort_values(by='song_name')

        # 4) Select TOP 10 only
        top_n = min(10, top_n)
        top_songs = df_lang.head(top_n)

        # Convert to list of dictionaries and ensure required fields
        recommendations = []
        for _, row in top_songs.iterrows():
            recommendations.append({
                'song_name': row.get('song_name'),
                'artist': row.get('artist'),
                'language': row.get('language'),
                'emotion': row.get('emotion'),
                'youtube_url': row.get('youtube_url') if 'youtube_url' in row.index else None,
                'spotify_url': row.get('spotify_url') if 'spotify_url' in row.index else None,
                'genre': row.get('genre', None),
                'recommendation_score': row.get('relevance_score') if 'relevance_score' in row.index else None
            })

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
