"""
Fine-tuning script for the emotion detection model.
Allows you to fine-tune the pre-trained model on custom data.
"""

import numpy as np
import cv2
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from emotion_detector import EmotionDetector

def prepare_training_data(data_dir):
    """
    Prepare training data from directory structure.
    Expected structure:
    data_dir/
        Angry/
            img1.jpg, img2.jpg, ...
        Happy/
            img1.jpg, img2.jpg, ...
        ...
    """
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("\nExpected structure:")
        print("data_dir/")
        print("  Angry/")
        print("  Happy/")
        print("  Sad/")
        print("  ...")
        return None
    
    # Create data generator with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def finetune_model(epochs=10, data_dir='training_data', save_path='models/emotion_model_finetuned.h5'):
    """
    Fine-tune the pre-trained emotion detection model.
    
    Args:
        epochs: Number of training epochs
        data_dir: Directory containing training data
        save_path: Path to save fine-tuned model
    """
    print("=" * 60)
    print("Fine-tuning Emotion Detection Model")
    print("=" * 60)
    
    # Load the detector (will load pre-trained model)
    detector = EmotionDetector()
    
    if detector.model is None:
        print("❌ Could not load model. Make sure pre-trained model exists.")
        return
    
    print(f"\n📊 Model Summary:")
    detector.model.summary()
    
    # Prepare training data
    print(f"\n📁 Loading training data from: {data_dir}")
    generators = prepare_training_data(data_dir)
    
    if generators is None:
        print("\n💡 To fine-tune the model:")
        print("1. Create a directory structure with emotion folders")
        print("2. Place training images in respective emotion folders")
        print("3. Run this script again")
        return
    
    train_gen, val_gen = generators
    
    print(f"\n✅ Found {train_gen.samples} training samples")
    print(f"✅ Found {val_gen.samples} validation samples")
    print(f"✅ Classes: {list(train_gen.class_indices.keys())}")
    
    # Set up callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Fine-tune the model
    print(f"\n🚀 Starting fine-tuning for {epochs} epochs...")
    print("   (This may take a while...)")
    
    history = detector.model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {save_path}")
    print(f"\n📈 Final Results:")
    print(f"   Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return detector.model

if __name__ == "__main__":
    import sys
    
    epochs = 10
    data_dir = 'training_data'
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    
    print("Fine-tuning Emotion Detection Model")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print()
    
    finetune_model(epochs=epochs, data_dir=data_dir)

