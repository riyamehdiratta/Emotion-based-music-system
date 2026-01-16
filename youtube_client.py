import os
import requests
from typing import List, Dict, Optional

YOUTUBE_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'
YOUTUBE_VIDEOS_URL = 'https://www.googleapis.com/youtube/v3/videos'


class YouTubeClient:
    """Simple wrapper around YouTube Data API v3 for deterministic language-filtered searches."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise RuntimeError('YOUTUBE_API_KEY not found in environment')

    def search_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'key': self.api_key,
            'videoEmbeddable': 'true'
        }
        r = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get('items', [])
        video_ids = [it['id']['videoId'] for it in items if it.get('id') and it['id'].get('videoId')]

        if not video_ids:
            return []

        # Fetch stats for these videos
        videos = self._get_videos(video_ids)
        # Preserve original order from search results
        id_to_item = {v['id']: v for v in videos}
        ordered = [id_to_item[vid] for vid in video_ids if vid in id_to_item]
        return ordered

    def _get_videos(self, video_ids: List[str]) -> List[Dict]:
        params = {
            'part': 'snippet,statistics,contentDetails',
            'id': ','.join(video_ids),
            'key': self.api_key,
            'maxResults': len(video_ids)
        }
        r = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get('items', [])
        videos = []
        for it in items:
            stats = it.get('statistics', {})
            snippet = it.get('snippet', {})
            videos.append({
                'id': it.get('id'),
                'title': snippet.get('title'),
                'channel_title': snippet.get('channelTitle'),
                'description': snippet.get('description'),
                'thumbnails': snippet.get('thumbnails', {}),
                'viewCount': int(stats.get('viewCount', 0)),
                'likeCount': int(stats.get('likeCount', 0)) if stats.get('likeCount') else 0,
                'commentCount': int(stats.get('commentCount', 0)) if stats.get('commentCount') else 0,
                'url': f'https://www.youtube.com/watch?v={it.get("id")}'
            })

        return videos
