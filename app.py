"""
Flask Application for Emotion-Based Music Recommendation System
Main backend server with API endpoints.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import numpy as np
import cv2
from dotenv import load_dotenv
from emotion_detector import EmotionDetector
from recommender import MusicRecommender

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize modules
# Emotion detector uses the shared preprocessing and model saved at models/emotion_model.h5
emotion_detector = EmotionDetector()
print("✅ Emotion detector initialized")

# Initialize recommender to use YouTube recommendations (deterministic language filtering)
try:
    music_recommender = MusicRecommender(use_spotify=False)
    print("✅ Music recommender initialized (YouTube mode)")
except Exception as e:
    print(f"⚠️ Recommender initialization failed: {e}")
    music_recommender = MusicRecommender(use_spotify=False)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/songs', exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_base64_image(image_data):
    """Decode base64 image string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    """
    API endpoint to detect emotion from uploaded image or base64 data.
    
    Expected JSON:
    {
        "image_data": "base64_string" (optional),
        "image_path": "path/to/image" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Check if image data is provided
        if 'image_data' in data and data['image_data']:
            # Decode base64 image
            image = decode_base64_image(data['image_data'])
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not decode image'
                }), 400
            
            result = emotion_detector.predict_emotion(image_array=image)
        
        elif 'image_path' in data and data['image_path']:
            result = emotion_detector.predict_emotion(image_path=data['image_path'])
        
        else:
            return jsonify({
                'success': False,
                'error': 'No image data or path provided'
            }), 400
        
        if not result.get('face_detected'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'No face detected in image'),
                'face_detected': False
            }), 400
        
        return jsonify({
            'success': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'all_emotions': result['all_emotions'],
            'face_detected': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to get music recommendations based on emotion.
    
    Expected JSON:
    {
        "emotion": "Happy",
        "confidence": 0.85,
        "language": "English" (optional),
        "top_n": 5 (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'emotion' not in data:
            return jsonify({
                'success': False,
                'error': 'Emotion not provided'
            }), 400
        
        emotion = data['emotion']
        confidence = data.get('confidence', 1.0)
        language = data.get('language') or 'English'
        top_n = data.get('top_n', 10)
        
        recommendations = music_recommender.get_recommendations(
            emotion=emotion,
            confidence=confidence,
            top_n=top_n,
            language=language
        )
        
        if not recommendations:
            return jsonify({
                'success': True,
                'recommendations': [],
                'count': 0,
                'message': 'No songs found for this emotion in selected language.'
            })

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detect-and-recommend', methods=['POST'])
def detect_and_recommend():
    """
    Combined endpoint: detect emotion and get recommendations in one call.
    
    Expected JSON:
    {
        "image_data": "base64_string" (optional),
        "image_path": "path/to/image" (optional),
        "language": "English" (optional),
        "top_n": 5 (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Detect emotion
        if 'image_data' in data and data['image_data']:
            image = decode_base64_image(data['image_data'])
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not decode image'
                }), 400
            emotion_result = emotion_detector.predict_emotion(image_array=image)
        
        elif 'image_path' in data and data['image_path']:
            emotion_result = emotion_detector.predict_emotion(image_path=data['image_path'])
        
        else:
            return jsonify({
                'success': False,
                'error': 'No image data or path provided'
            }), 400
        
        if not emotion_result.get('face_detected'):
            return jsonify({
                'success': False,
                'error': emotion_result.get('error', 'No face detected in image'),
                'face_detected': False
            }), 400
        
        # Get recommendations
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        language = data.get('language') or 'English'
        top_n = data.get('top_n', 10)
        
        recommendations = music_recommender.get_recommendations(
            emotion=emotion,
            confidence=confidence,
            top_n=top_n,
            language=language
        )
        
        if not recommendations:
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': emotion_result['all_emotions'],
                'recommendations': [],
                'count': 0,
                'message': 'No songs found for this emotion in selected language.'
            })

        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': emotion_result['all_emotions'],
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get list of all available languages."""
    try:
        # Provide a strict language selector as required
        languages = ['English', 'Hindi', 'Punjabi', 'Spanish']
        return jsonify({
            'success': True,
            'languages': languages
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the song database."""
    try:
        stats = {
            'song_count_by_emotion': music_recommender.get_song_count_by_emotion(),
            'languages': music_recommender.get_all_languages(),
            'total_songs': len(music_recommender.songs_df) if music_recommender.songs_df is not None else 0
        }
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/songs/<path:filename>')
def serve_song(filename):
    """Serve audio files from the songs directory."""
    return send_from_directory('static/songs', filename)

if __name__ == '__main__':
    print("=" * 60)
    print("Emotion-Based Music Recommendation System")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nNote: This system recommends music, not medical advice.")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)

