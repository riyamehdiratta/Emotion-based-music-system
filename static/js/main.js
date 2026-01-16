// Emotion-Based Music Recommendation System - Main JavaScript

let currentImageData = null;
let currentEmotion = null;
let currentConfidence = null;
let webcamStream = null;
let selectedLanguage = '';

// Emotion icon mapping
const emotionIcons = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Neutral': '😐',
    'Surprise': '😲',
    'Fear': '😨',
    'Disgust': '🤢'
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    loadLanguages();
});

// Initialize image upload
function initializeUpload() {
    const uploadArea = document.getElementById('upload-area');
    const imageInput = document.getElementById('image-input');
    
    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    
    imageInput.addEventListener('change', handleImageSelect);
}

// Handle drag and drop
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.style.background = '#f0f0ff';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.style.background = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    if (file.size > 5 * 1024 * 1024) {
        alert('File size must be less than 5MB');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        displayImagePreview(e.target.result);
        currentImageData = e.target.result;
        document.getElementById('detect-btn').disabled = false;
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(imageSrc) {
    const preview = document.getElementById('image-preview');
    const img = document.getElementById('preview-img');
    const uploadArea = document.getElementById('upload-area');
    
    img.src = imageSrc;
    preview.style.display = 'block';
    uploadArea.style.display = 'none';
}

function removeImage() {
    currentImageData = null;
    document.getElementById('image-preview').style.display = 'none';
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('image-input').value = '';
    document.getElementById('detect-btn').disabled = true;
}

// Tab switching
function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    if (tab === 'upload') {
        document.getElementById('upload-tab').classList.add('active');
        stopWebcam();
    } else {
        document.getElementById('webcam-tab').classList.add('active');
    }
}

// Webcam functions
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        webcamStream = stream;
        const video = document.getElementById('webcam-video');
        video.srcObject = stream;
        video.style.display = 'block';
        
        document.getElementById('webcam-placeholder').style.display = 'none';
        document.getElementById('start-webcam-btn').style.display = 'none';
        document.getElementById('capture-btn').style.display = 'inline-block';
        document.getElementById('stop-webcam-btn').style.display = 'inline-block';
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam. Please check permissions.');
    }
}

function captureWebcam() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    currentImageData = canvas.toDataURL('image/jpeg');
    document.getElementById('detect-btn').disabled = false;
    
    // Show preview
    displayImagePreview(currentImageData);
    
    alert('Photo captured! Click "Detect Emotion" to analyze.');
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    const video = document.getElementById('webcam-video');
    video.srcObject = null;
    video.style.display = 'none';
    
    document.getElementById('webcam-placeholder').style.display = 'block';
    document.getElementById('start-webcam-btn').style.display = 'inline-block';
    document.getElementById('capture-btn').style.display = 'none';
    document.getElementById('stop-webcam-btn').style.display = 'none';
}

// Detect emotion
async function detectEmotion() {
    if (!currentImageData) {
        alert('Please select or capture an image first');
        return;
    }
    
    const detectBtn = document.getElementById('detect-btn');
    detectBtn.disabled = true;
    detectBtn.innerHTML = '<span class="loading"></span> Detecting...';
    
    try {
        const response = await fetch('/api/detect-emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_data: currentImageData
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayEmotionResult(data);
            // Do not fetch recommendations yet — user must select language
        } else {
            alert('Error: ' + (data.error || 'Could not detect emotion'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        detectBtn.disabled = false;
        detectBtn.innerHTML = '🔍 Detect Emotion';
    }
}

// Display emotion result
function displayEmotionResult(data) {
    currentEmotion = data.emotion;
    currentConfidence = data.confidence;
    
    // Update emotion display
    document.getElementById('emotion-icon').textContent = emotionIcons[data.emotion] || '😐';
    document.getElementById('emotion-name').textContent = data.emotion;
    document.getElementById('confidence-score').textContent = (data.confidence * 100).toFixed(1) + '%';
    
    // Display all emotions
    const allEmotionsDiv = document.getElementById('all-emotions');
    allEmotionsDiv.innerHTML = '';
    
    Object.entries(data.all_emotions).forEach(([emotion, score]) => {
        const emotionBar = document.createElement('div');
        emotionBar.className = 'emotion-bar';
        emotionBar.innerHTML = `
            <div class="emotion-bar-label">${emotion} ${emotionIcons[emotion] || ''}</div>
            <div class="emotion-bar-fill" style="width: ${score * 100}%"></div>
            <div style="margin-top: 5px; font-size: 0.9em; color: #666;">${(score * 100).toFixed(1)}%</div>
        `;
        allEmotionsDiv.appendChild(emotionBar);
    });
    
    // Show result section
    document.getElementById('emotion-result').style.display = 'block';
    document.getElementById('language-section').style.display = 'block';
    document.getElementById('recommendations-section').style.display = 'block';
}

// Display recommendations
function displayRecommendations(recommendations) {
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = '';
    
    if (recommendations.length === 0) {
        recommendationsList.innerHTML = '<p>No recommendations found. Try a different emotion or language.</p>';
        return;
    }
    
    recommendations.forEach((song, index) => {
        const songCard = document.createElement('div');
        songCard.className = 'song-card';
        
        // Check if this is a Spotify track
        const isSpotify = song.spotify_url || song.spotify_id;
        const isYouTube = song.youtube_url || song.youtube_id;
        
        // Album art or placeholder
        const albumArt = song.album_art || 'https://via.placeholder.com/150?text=No+Image';
        
        songCard.innerHTML = `
            <div class="song-number">#${index + 1}</div>
            <div class="song-album-art">
                <img src="${albumArt}" alt="${song.song_name}" onerror="this.src='https://via.placeholder.com/150?text=No+Image'">
            </div>
            <div class="song-info">
                <div class="song-title">${song.song_name}</div>
                <div class="song-artist">🎤 ${song.artist}</div>
                <div class="song-meta">
                    <span class="meta-badge badge-language">🌍 ${song.language}</span>
                    <span class="meta-badge badge-emotion">${emotionIcons[song.emotion] || ''} ${song.emotion}</span>
                    <span class="meta-badge badge-genre">🎵 ${song.genre || 'Unknown'}</span>
                </div>
                ${isSpotify ? `
                    <div class="spotify-actions">
                        <a href="${song.spotify_url}" target="_blank" class="spotify-button">
                            <span style="color: #1DB954;">♫</span> Open in Spotify
                        </a>
                    </div>
                ` : ''}
            </div>
            <div class="audio-player">
                ${isYouTube && song.youtube_id ?
                    `<iframe width="100%" height="200" src="https://www.youtube.com/embed/${song.youtube_id}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`
                    : isSpotify && song.spotify_id ?
                    `<iframe 
                        src="https://open.spotify.com/embed/track/${song.spotify_id}?utm_source=generator&theme=0" 
                        width="100%" 
                        height="152" 
                        frameBorder="0" 
                        allowfullscreen="" 
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                        loading="lazy">
                    </iframe>`
                    : song.audio_path ?
                    `<audio controls>
                        <source src="/${song.audio_path}" type="audio/mpeg">
                        Your browser does not support the audio element.
                    </audio>`
                    : '<p style="color: #999; font-size: 0.9em;">Preview not available</p>'
                }
            </div>
        `;
        
        recommendationsList.appendChild(songCard);
    });
    
    // Add playlist embed if all tracks are from same playlist
    const playlistIds = [...new Set(recommendations.filter(s => s.playlist_id).map(s => s.playlist_id))];
    if (playlistIds.length === 1 && playlistIds[0]) {
        const playlistEmbed = document.createElement('div');
        playlistEmbed.className = 'playlist-embed-container';
        playlistEmbed.innerHTML = `
            <h3 style="margin: 20px 0 10px 0; color: #667eea;">🎵 Full Playlist</h3>
            <iframe
                title="Spotify Embed: Recommendation Playlist"
                src="https://open.spotify.com/embed/playlist/${playlistIds[0]}?utm_source=generator&theme=0"
                width="100%"
                height="360"
                style="min-height: 360px; border-radius: 12px;"
                frameBorder="0"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy">
            </iframe>
        `;
        recommendationsList.appendChild(playlistEmbed);
    }
    
    // Add Spotify attribution
    if (recommendations.some(s => s.spotify_url)) {
        const attribution = document.createElement('div');
        attribution.className = 'spotify-attribution';
        attribution.innerHTML = '<p>Powered by <a href="https://www.spotify.com" target="_blank">Spotify</a></p>';
        recommendationsList.appendChild(attribution);
    }
}

// Load available languages
async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        const data = await response.json();
        
        if (data.success) {
                const languageFilter = document.getElementById('language-filter');
                // Populate language dropdown that appears AFTER detection
                data.languages.forEach(lang => {
                    const option = document.createElement('option');
                    option.value = lang;
                    option.textContent = lang;
                    if (languageFilter) languageFilter.appendChild(option.cloneNode(true));
                });
                // Default to English when available
                if (languageFilter) {
                    languageFilter.value = 'English';
                    selectedLanguage = 'English';
                }
        }
    } catch (error) {
        console.error('Error loading languages:', error);
    }
}

// Filter by language
function filterByLanguage() {
    selectedLanguage = document.getElementById('language-filter').value;
    
    if (currentEmotion && currentConfidence) {
        // Re-fetch recommendations with language filter
        fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                emotion: currentEmotion,
                confidence: currentConfidence,
                language: selectedLanguage || null,
                top_n: 5
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayRecommendations(data.recommendations);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}

function onLanguageSelect() {
    const v = document.getElementById('language-filter').value;
    selectedLanguage = v;
    const disp = document.getElementById('selected-language');
    if (disp) disp.textContent = v || 'None';
}

