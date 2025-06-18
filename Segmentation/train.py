import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import unet_model
from preprocess import load_image_mask_pairs

# --- CONFIG ---
IMAGE_DIR = 'data/segmentation/images/'
MASK_DIR = 'data/segmentation/masks/'
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

# --- Load Data ---
images, masks = load_image_mask_pairs(IMAGE_DIR, MASK_DIR, IMG_SIZE)
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# --- Create Dataset ---
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Build Model ---
model = unet_model(input_size=(*IMG_SIZE, 3), num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Train ---
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# --- Save Model ---
model.save("segmentation_model.h5")
print("✅ Model saved to segmentation_model.h5")
print("✅ Training complete!")