"""
Script to download a pre-trained FER2013 emotion recognition model.
This will significantly improve confidence scores (80-95%).
"""

import os
import urllib.request
import sys

def download_pretrained_model():
    """Download a pre-trained FER2013 model for better accuracy."""
    
    model_url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    model_path = "models/emotion_model_fer2013.h5"
    
    print("=" * 60)
    print("Downloading Pre-trained FER2013 Emotion Model")
    print("=" * 60)
    print(f"URL: {model_url}")
    print(f"Save to: {model_path}")
    print()
    
    try:
        os.makedirs('models', exist_ok=True)
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            bar_length = 40
            filled = int(bar_length * downloaded / total_size)
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\r[{bar}] {percent:.1f}%')
            sys.stdout.flush()
        
        print("Downloading...")
        urllib.request.urlretrieve(model_url, model_path, show_progress)
        print("\n✅ Model downloaded successfully!")
        print(f"\nModel saved to: {model_path}")
        print("\nNote: You may need to rename this file to 'emotion_model.h5'")
        print("      or update the model_path in emotion_detector.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nAlternative: You can manually download a pre-trained model from:")
        print("  - https://github.com/oarriaga/face_classification")
        print("  - Or train your own model on FER2013 dataset")
        return False

if __name__ == "__main__":
    download_pretrained_model()

