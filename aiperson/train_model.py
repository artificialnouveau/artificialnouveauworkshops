#!/usr/bin/env python3
"""
Train a Pro AI vs Anti AI image classifier using MobileNetV2 transfer learning.
Exports the model to TensorFlow.js format for browser-based inference.
"""

import os
import numpy as np
from pathlib import Path

# Register HEIF/AVIF support before any image loading
import pillow_heif
pillow_heif.register_heif_opener()

from PIL import Image, ImageEnhance
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Config
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
AUGMENT_FACTOR = 5  # Generate this many augmented copies per image
BASE_DIR = Path(__file__).parent


def augment_image(img):
    """Apply random augmentations to a PIL image."""
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    # Random brightness
    factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)
    # Random zoom (crop and resize)
    if random.random() > 0.5:
        w, h = img.size
        crop_frac = random.uniform(0.85, 1.0)
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)
    return img


def load_images_from_folder(folder, label, augment=True):
    """Load and preprocess images from a folder, handling multiple formats."""
    images = []
    labels = []
    supported_exts = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.avif'}

    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in supported_exts:
            continue
        try:
            img = Image.open(f).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

            # Original image
            arr = np.array(img, dtype=np.float32)
            images.append(arr)
            labels.append(label)

            # Augmented copies
            if augment:
                for _ in range(AUGMENT_FACTOR):
                    aug_img = augment_image(img.copy())
                    arr = np.array(aug_img, dtype=np.float32)
                    images.append(arr)
                    labels.append(label)

            print(f"  Loaded: {f.name}")
        except Exception as e:
            print(f"  SKIP: {f.name} ({e})")

    return images, labels


def main():
    pro_dir = BASE_DIR / "pro ai"
    anti_dir = BASE_DIR / "anti ai"

    print("Loading Pro AI images...")
    pro_imgs, pro_labels = load_images_from_folder(pro_dir, 1.0)  # 1 = pro AI
    print(f"  → {len(pro_imgs)} images (with augmentation)\n")

    print("Loading Anti AI images...")
    anti_imgs, anti_labels = load_images_from_folder(anti_dir, 0.0)  # 0 = anti AI
    print(f"  → {len(anti_imgs)} images (with augmentation)\n")

    X = np.array(pro_imgs + anti_imgs)
    y = np.array(pro_labels + anti_labels)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Preprocess for MobileNetV2 (scales to [-1, 1])
    X = preprocess_input(X)

    print(f"Total dataset: {len(X)} images ({int(y.sum())} pro, {len(y) - int(y.sum())} anti)\n")

    # Build model (no augmentation layers - augmentation done in preprocessing)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train
    print("\nTraining...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )

    val_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\nBest validation accuracy: {val_acc:.2%}")

    # Save as H5 format for tfjs conversion
    h5_path = str(BASE_DIR / "model.h5")
    model.save(h5_path)
    print(f"Saved H5 model to {h5_path}")

    # Convert to TensorFlow.js using command-line converter
    tfjs_path = str(BASE_DIR / "tfjs_model")
    import subprocess, shutil
    if Path(tfjs_path).exists():
        shutil.rmtree(tfjs_path)
    result = subprocess.run([
        "tensorflowjs_converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        h5_path,
        tfjs_path
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion error: {result.stderr}")
    else:
        print(f"Exported TF.js model to {tfjs_path}/")

    # List output files
    for f in sorted(Path(tfjs_path).iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
