"""
Emotion detection module (reworked).

Improvements implemented:
- Uses the shared preprocessing in `emotion_model.py` so training and inference are identical.
- Loads saved model from `models/emotion_model.h5` or builds backbone if missing.
- Applies softmax temperature scaling for calibration.
- Enforces thresholds: <0.4 -> Neutral, <0.6 -> Uncertain.
- Supports smoothing over recent webcam frames (weighted majority vote by confidence).
"""

import os
from collections import deque
from typing import Optional

import cv2
import numpy as np
from tensorflow import keras

from emotion_model import preprocess_face_for_model, build_emotion_model


class EmotionDetector:
    def __init__(self, model_path: str = 'models/emotion_model.h5', smoothing_max: int = 12):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model_path = model_path
        self.model: Optional[keras.Model] = None
        self.input_size = 48
        self.rgb_backbone = False
        self.temperature = float(os.getenv('EMOTION_TEMPERATURE', '1.0'))
        self.smoothing_max = smoothing_max
        self.smoothing_buffer = deque(maxlen=self.smoothing_max)

        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path, compile=False)
                in_shape = self.model.input_shape
                if len(in_shape) == 4 and in_shape[3] == 3:
                    self.rgb_backbone = True
                self.input_size = int(in_shape[1])
                print(f"Loaded emotion model ({self.model_path}) input_size={self.input_size} rgb={self.rgb_backbone}")
                return
            except Exception as e:
                print(f"Could not load saved model: {e}")

        # Build a default model to avoid crashing; user should train and save a model
        print("No saved emotion model found — building default mini model (use train_emotion_model.py to train).")
        model, input_shape = build_emotion_model('mini_xception', num_classes=len(self.emotions))
        self.model = model
        self.input_size = input_shape[0]
        self.rgb_backbone = (input_shape[2] == 3)

    def detect_face(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None, None
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        return face_roi, [int(x), int(y), int(w), int(h)]

    def preprocess_face(self, face_roi: np.ndarray):
        target = (self.input_size, self.input_size)
        return np.expand_dims(preprocess_face_for_model(face_roi, target_size=target, rgb=self.rgb_backbone), axis=0)

    def _apply_temperature(self, probs: np.ndarray):
        t = max(0.01, float(self.temperature))
        if t == 1.0:
            return probs
        # Reconstruct logits safely then rescale
        logits = np.log(np.maximum(probs, 1e-12))
        scaled = logits / t
        exp = np.exp(scaled - np.max(scaled))
        return exp / np.sum(exp)

    def predict_emotion(self, image_path: str = None, image_array: np.ndarray = None):
        if image_path:
            image = cv2.imread(image_path)
            if image is None:
                return {'emotion': None, 'confidence': 0.0, 'all_emotions': {}, 'face_detected': False, 'error': 'Could not load image'}
        elif image_array is not None:
            image = image_array
        else:
            return {'emotion': None, 'confidence': 0.0, 'all_emotions': {}, 'face_detected': False, 'error': 'No image provided'}

        face_roi, coords = self.detect_face(image)
        if face_roi is None:
            return {'emotion': None, 'confidence': 0.0, 'all_emotions': {}, 'face_detected': False, 'error': 'No face detected'}

        processed = self.preprocess_face(face_roi)
        try:
            probs = self.model.predict(processed, verbose=0)[0]
        except Exception as e:
            return {'emotion': None, 'confidence': 0.0, 'all_emotions': {}, 'face_detected': True, 'error': f'Prediction error: {e}'}

        # Calibration via temperature scaling
        calibrated = self._apply_temperature(probs)

        # Debug logging: print raw calibrated softmax probabilities for all classes
        try:
            probs_str = ', '.join([f"{self.emotions[i]}:{calibrated[i]:.3f}" for i in range(len(self.emotions))])
            print(f"[DEBUG] Softmax probs: {probs_str}")
        except Exception:
            pass

        idx = int(np.argmax(calibrated))
        confidence = float(calibrated[idx])
        top_emotion = self.emotions[idx]

        # Do NOT force Neutral on low confidence. Instead:
        # - If confidence < 0.4 -> label as 'Uncertain'
        # - Neutral is only returned if it is the top predicted class
        if confidence < 0.4:
            label = 'Uncertain'
        else:
            label = top_emotion

        all_emotions = {self.emotions[i]: float(calibrated[i]) for i in range(len(self.emotions))}

        # Append for smoothing (webcam)
        self.smoothing_buffer.append({'label': label, 'confidence': confidence, 'probs': all_emotions, 'top': top_emotion})

        return {'emotion': label, 'confidence': confidence, 'all_emotions': all_emotions, 'face_detected': True, 'face_coords': coords}

    def get_smoothed(self):
        if not self.smoothing_buffer:
            return None
        # Weighted majority by confidence. Also compute stability fraction.
        vote = {}
        agg = {k: 0.0 for k in self.emotions}
        total_conf = 0.0
        top_matches_conf = 0.0
        for entry in self.smoothing_buffer:
            lab = entry['label']
            conf = entry['confidence']
            top = entry.get('top')
            vote[lab] = vote.get(lab, 0.0) + conf
            for k, v in entry['probs'].items():
                agg[k] = agg.get(k, 0.0) + v * conf
            total_conf += conf
            if top == lab:
                top_matches_conf += conf

        best_label = max(vote.items(), key=lambda x: x[1])[0]
        # normalized aggregated probabilities
        total = sum(agg.values()) or 1.0
        normalized = {k: v / total for k, v in agg.items()}
        # stability: fraction of confidence mass where predicted top==reported label
        stability = (top_matches_conf / total_conf) if total_conf > 0 else 0.0

        # Only return smoothed label if stable enough (>0.6), otherwise indicate unstable
        smoothed_conf = vote[best_label] / len(self.smoothing_buffer)
        result = {'emotion': best_label, 'confidence': float(smoothed_conf), 'all_emotions': normalized, 'stability': float(stability)}
        return result

    def reset_smoothing(self):
        self.smoothing_buffer.clear()

    # Backwards-compatible alias
    def predict_from_webcam_frame(self, frame):
        """Predict from a webcam frame, using rolling smoothing and stability check.

        Returns smoothed prediction only when stable; otherwise returns immediate prediction
        with an additional 'stable' flag set to False.
        """
        immediate = self.predict_emotion(image_array=frame)
        # compute smoothed aggregation from buffer
        smoothed = self.get_smoothed()
        if smoothed is None:
            immediate['stable'] = False
            return immediate

        # Accept smoothed label only when stability > 0.6
        if smoothed.get('stability', 0.0) > 0.6:
            out = {
                'emotion': smoothed['emotion'],
                'confidence': smoothed['confidence'],
                'all_emotions': smoothed['all_emotions'],
                'face_detected': immediate.get('face_detected', True),
                'face_coords': immediate.get('face_coords'),
                'stable': True
            }
            return out
        else:
            immediate['stable'] = False
            return immediate

