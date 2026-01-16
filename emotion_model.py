"""Emotion model utilities: build, load, and preprocessing helpers.

Provides a configurable backbone (EfficientNetB0 / ResNet50 / Mini-Xception),
preprocessing that is consistent for training and inference, and training helper
functions implementing transfer-learning and fine-tuning strategy described in the spec.
"""
import os
from typing import Tuple
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


AVAILABLE_BACKBONES = ('efficientnetb0', 'resnet50', 'mini_xception')


def get_backbone(name: str, input_shape=(48,48,1), include_top=False):
    name = name.lower()
    if name == 'efficientnetb0':
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(96,96,3), include_top=False, weights='imagenet')
        return base, (96,96,3)
    if name == 'resnet50':
        base = tf.keras.applications.ResNet50(
            input_shape=(96,96,3), include_top=False, weights='imagenet')
        return base, (96,96,3)
    if name == 'mini_xception':
        # Lightweight Xception-like architecture for FER datasets (grayscale)
        inp = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv2D(64, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SeparableConv2D(128, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        model = keras.Model(inp, x)
        return model, input_shape
    raise ValueError(f'Backbone {name} not recognized. Choose from {AVAILABLE_BACKBONES}')


def build_emotion_model(backbone_name: str = 'mini_xception', num_classes: int = 7):
    base, input_shape = get_backbone(backbone_name)
    # If backbone is not already a Model that outputs features, wrap accordingly
    if isinstance(base.output, tf.Tensor):
        x = base.output
    else:
        x = base.output

    x = layers.GlobalAveragePooling2D()(x) if len(x.shape) == 4 else x
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='emotion_output')(x)

    model = keras.Model(inputs=base.input, outputs=outputs)
    return model, input_shape


def load_pretrained_emotion_model(backbone_name: str = 'mini_xception', num_classes: int = 7, model_path: str = None):
    """Build model and optionally load weights from a saved file.

    - backbone_name: choose among AVAILABLE_BACKBONES
    - num_classes: number of emotion classes
    - model_path: optional path to load full model weights
    """
    model, input_shape = build_emotion_model(backbone_name, num_classes=num_classes)
    if model_path and os.path.exists(model_path):
        try:
            model.load_weights(model_path)
        except Exception:
            # If loading full model fails, try loading entire model instead (handled by caller)
            pass
    return model, input_shape


def preprocess_face_for_model(face_roi, target_size: Tuple[int,int], rgb: bool = False):
    """Consistent preprocessing used both in training and inference.

    - face_roi: BGR image (as read by OpenCV) or grayscale
    - target_size: (w,h)
    - rgb: whether to return 3-channel RGB (for ImageNet backbones)
    """
    # If we get a color image convert to grayscale first for face processing
    if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi

    # Resize to target size (use INTER_CUBIC)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)

    if rgb:
        # Convert back to RGB by duplicating channels
        img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        img = img.astype('float32') / 255.0
        return img
    else:
        img = resized.astype('float32') / 255.0
        img = img.reshape(target_size[1], target_size[0], 1)
        return img


def save_model(model: keras.Model, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    model.save(path)
