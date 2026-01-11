# 🚀 Quick Start Guide

## Installation (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Setup (Optional)
```bash
bash setup.sh
```

Or manually create directories:
```bash
mkdir -p static/songs/{english,hindi,punjabi,tamil,telugu,korean,spanish} models uploads
```

### Step 3: Start the Server
```bash
python app.py
```

## First Run

1. **Open Browser**: Navigate to `http://localhost:5000`

2. **The system will automatically**:
   - Create `songs_metadata.csv` with sample songs
   - Create a simple emotion detection model (if not present)

3. **To use the system**:
   - Upload an image with a face OR use webcam
   - Click "Detect Emotion"
   - View recommendations

## Adding Your Own Songs

1. **Add audio files** to language folders:
   ```
   static/songs/english/my_song.mp3
   ```

2. **Update metadata** in `songs_metadata.csv`:
   ```csv
   song_name,artist,language,emotion,genre,audio_path
   "My Song","Artist Name","English","Happy","Pop","static/songs/english/my_song.mp3"
   ```

## Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Webcam Not Working
- Use HTTPS in production
- Check browser permissions
- Try different browser

## Next Steps

- Add your favorite songs to the database
- Train a better emotion model using FER2013 dataset
- Customize the UI in `static/css/style.css`
- Add more languages by creating new folders

---

**Ready to go!** 🎵

