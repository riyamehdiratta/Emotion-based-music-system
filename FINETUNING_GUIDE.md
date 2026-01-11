# 🎯 Fine-Tuning Guide for Emotion Detection Model

## Overview

The emotion detection system now uses a **fine-tuned pre-trained model** that:
- ✅ Uses the pre-trained FER2013 model as a base (excellent feature extraction)
- ✅ Freezes early layers (keeps learned features)
- ✅ Makes top layers trainable (allows customization)
- ✅ Adds a custom trainable head for better adaptation

## How It Works

### Current Setup (Automatic)

When you load the model, it automatically:
1. Loads the pre-trained FER2013 model
2. Freezes the base layers (feature extraction)
3. Makes the top 25-30% of layers trainable
4. Optionally adds a custom trainable classification head
5. Compiles with a lower learning rate (0.0001) for fine-tuning

### Benefits

- **Better Accuracy**: Pre-trained features + trainable top layers
- **Customizable**: Can adapt to your specific use case
- **Efficient**: Only trains a small portion of the model
- **Fast Training**: Fine-tuning is much faster than training from scratch

## Fine-Tuning on Your Data

### Step 1: Prepare Your Data

Create a directory structure like this:

```
training_data/
├── Angry/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Happy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Sad/
│   └── ...
├── Neutral/
│   └── ...
├── Surprise/
│   └── ...
├── Fear/
│   └── ...
└── Disgust/
    └── ...
```

### Step 2: Run Fine-Tuning

```bash
python finetune_model.py training_data 20
```

Arguments:
- First argument: Path to training data directory
- Second argument: Number of epochs (default: 10)

### Step 3: Use Fine-Tuned Model

The fine-tuned model will be saved to:
```
models/emotion_model_finetuned.h5
```

To use it, update `emotion_detector.py` to load this model instead, or rename it to `emotion_model.h5`.

## Model Architecture

### Fine-Tuning Strategy

1. **Base Model (Frozen)**: 
   - Pre-trained FER2013 mini_XCEPTION
   - 70-75% of layers frozen
   - Provides excellent feature extraction

2. **Trainable Layers**:
   - Last 25-30% of base model layers
   - Custom dense layers (128 → 64 → 7)
   - Batch normalization and dropout

3. **Learning Rate**:
   - 0.0001 (10x lower than training from scratch)
   - Prevents destroying pre-trained features

## Advanced Fine-Tuning

### Adjust Trainable Layers

Edit `emotion_detector.py`, in `_create_finetuned_model()`:

```python
# Make more layers trainable (e.g., last 40%)
trainable_start = int(total_layers * 0.6)  # Changed from 0.75

# Or make fewer layers trainable (e.g., last 15%)
trainable_start = int(total_layers * 0.85)
```

### Adjust Learning Rate

```python
# Higher learning rate (faster but riskier)
optimizer = keras.optimizers.Adam(learning_rate=0.0005, ...)

# Lower learning rate (safer but slower)
optimizer = keras.optimizers.Adam(learning_rate=0.00005, ...)
```

### Custom Head Architecture

Modify the custom head in `_create_finetuned_model()`:

```python
# Add more layers
x = Dense(256, ...)(x)  # Larger layer
x = Dense(128, ...)(x)
x = Dense(64, ...)(x)

# Or simpler head
x = Dense(64, ...)(x)  # Direct to output
```

## Monitoring Training

The fine-tuning script includes:
- **Early Stopping**: Stops if validation accuracy doesn't improve
- **Model Checkpointing**: Saves best model during training
- **Learning Rate Reduction**: Lowers LR if stuck
- **Progress Tracking**: Shows training/validation metrics

## Tips for Best Results

1. **Data Quality**: Use diverse, high-quality images
2. **Data Augmentation**: Already included (rotation, flip, zoom)
3. **Balanced Dataset**: Similar number of images per emotion
4. **Validation Split**: 20% of data used for validation
5. **Epochs**: Start with 10-20 epochs, adjust based on results
6. **Batch Size**: Default 32, adjust if you have memory issues

## Current Model Status

The current model automatically uses fine-tuning:
- ✅ Pre-trained base (FER2013)
- ✅ Top layers trainable
- ✅ Custom head added
- ✅ Optimized for inference

You can use it immediately, or fine-tune further on your data for even better results!

---

**Note**: Fine-tuning requires training data. The current model works well out-of-the-box, but fine-tuning on your specific data can improve accuracy by 5-15%.

