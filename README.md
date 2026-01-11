# 🎵 Emotion-Based Music Recommendation System

A sophisticated web application that detects emotions from facial images and recommends music that matches your mood across multiple languages.

## 🌟 Features

- **Emotion Detection**: CNN-based emotion recognition supporting 7 emotions:
  - Happy 😊
  - Sad 😢
  - Angry 😠
  - Neutral 😐
  - Surprise 😲
  - Fear 😨
  - Disgust 🤢

- **Multi-Language Support**: 
  - English
  - Hindi
  - Punjabi
  - Tamil
  - Telugu
  - Korean
  - Spanish
  - (Easily extensible)

- **Multiple Input Methods**:
  - Image upload (drag & drop or click)
  - Webcam capture

- **Smart Recommendations**:
  - Emotion-based filtering
  - Confidence score ranking
  - Language filtering
  - Top 5 recommendations

- **Beautiful UI**:
  - Modern, responsive design
  - Real-time emotion visualization
  - Audio player for each song
  - Artist names and metadata display

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (optional, for webcam feature)

## 🚀 Installation

1. **Clone or download this repository**

2. **Navigate to the project directory**
   ```bash
   cd winterhack
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create necessary directories**
   ```bash
   mkdir -p static/songs/english static/songs/hindi static/songs/punjabi static/songs/tamil static/songs/telugu static/songs/korean static/songs/spanish models uploads
   ```

## 🎯 Usage

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your web browser**
   Navigate to: `http://localhost:5000`

3. **Use the application**:
   - **Upload Image**: Click or drag an image with a face
   - **Webcam**: Click "Start Webcam" and capture a photo
   - Click "Detect Emotion" to analyze
   - View your detected emotion and confidence score
   - Browse recommended songs matching your mood
   - Filter by language (optional)
   - Play songs using the built-in audio player

## 📁 Project Structure

```
winterhack/
├── app.py                 # Flask backend application
├── emotion_detector.py    # Emotion detection module
├── recommender.py         # Music recommendation engine
├── songs_metadata.csv     # Song database (auto-generated)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   ├── css/
│   │   └── style.css     # Stylesheet
│   ├── js/
│   │   └── main.js       # Frontend JavaScript
│   └── songs/            # Audio files organized by language
│       ├── english/
│       ├── hindi/
│       ├── punjabi/
│       ├── tamil/
│       ├── telugu/
│       ├── korean/
│       └── spanish/
├── models/               # ML model storage
└── uploads/              # Temporary image uploads
```

## 🎵 Adding Your Own Songs

1. **Add audio files** to the appropriate language folder:
   ```
   static/songs/[language]/your_song.mp3
   ```

2. **Update `songs_metadata.csv`** with the following format:
   ```csv
   song_name,artist,language,emotion,genre,audio_path
   "Your Song Name","Artist Name","Language","Emotion","Genre","static/songs/language/your_song.mp3"
   ```

   Example:
   ```csv
   song_name,artist,language,emotion,genre,audio_path
   "Shape of You","Ed Sheeran","English","Happy","Pop","static/songs/english/shape_of_you.mp3"
   ```

## 🔧 Configuration

### Supported Emotions
- Happy
- Sad
- Angry
- Neutral
- Surprise
- Fear
- Disgust

### Supported Languages
- English
- Hindi
- Punjabi
- Tamil
- Telugu
- Korean
- Spanish

To add more languages, simply:
1. Create a new folder: `static/songs/[new_language]/`
2. Add songs to the metadata CSV with the new language name

## 🧠 How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade to detect faces in images
2. **Emotion Recognition**: CNN model processes the face region to predict emotion
3. **Music Matching**: Recommendation engine filters songs by detected emotion
4. **Ranking**: Songs are ranked by emotion confidence and mood match score
5. **Display**: Top 5 recommendations are shown with metadata and audio players

## ⚠️ Important Notes

- **No Medical Advice**: This system is for entertainment and music discovery only. It does not provide medical or psychological advice.
- **Privacy**: No emotion data is stored. All processing happens in real-time.
- **Model Accuracy**: The system uses an improved deep CNN architecture. For best results (80-95% confidence), install the FER library: `pip install fer`
- **Audio Files**: Sample metadata is provided, but you'll need to add your own audio files to the `static/songs/` directories.

## 🎯 Improving Model Accuracy

For **high confidence scores (80-95%)**, you have two options:

### Option 1: Use FER Library (Recommended)
```bash
pip install fer
```
The system will automatically use the pre-trained FER model which provides excellent accuracy.

### Option 2: Download Pre-trained Model
```bash
python download_model.py
```
This downloads a pre-trained FER2013 model for better accuracy.

## 🐛 Troubleshooting

### "No face detected"
- Ensure the image contains a clear, front-facing face
- Try a different image with better lighting
- Make sure the face is not too small or too large

### "Model not found"
- The system will create a simple model automatically
- For better accuracy, download a pre-trained FER2013 model and place it in `models/emotion_model.h5`

### "Audio not playing"
- Ensure audio files exist in the correct paths
- Check browser console for errors
- Verify file formats (MP3 recommended)

### Webcam not working
- Check browser permissions for camera access
- Use HTTPS in production (required for webcam)
- Try a different browser

## 🔒 Privacy & Ethics

- **No Data Storage**: Emotion detection happens in real-time; no data is saved
- **Local Processing**: All processing happens on your server
- **User Consent**: Users are informed this is for entertainment only
- **No Tracking**: No user tracking or analytics

## 🛠️ Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

### API Endpoints

- `GET /` - Main page
- `POST /api/detect-emotion` - Detect emotion from image
- `POST /api/recommend` - Get recommendations for emotion
- `POST /api/detect-and-recommend` - Combined detection and recommendation
- `GET /api/languages` - Get available languages
- `GET /api/stats` - Get database statistics
- `GET /songs/<path:filename>` - Serve audio files

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to:
- Add more songs to the database
- Improve the emotion detection model
- Add new languages
- Enhance the UI/UX
- Report bugs or suggest features

## 📧 Support

For issues or questions, please check the troubleshooting section or review the code comments.

---

**Built with ❤️ for music lovers**

*Remember: This system recommends music, not medical advice.*

