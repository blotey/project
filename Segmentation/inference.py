import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from preprocess import preprocess_image

# --- Load Model ---
model = load_model("segmentation_model.h5", compile=False)

# --- Load and Preprocess Image ---
if len(sys.argv) != 2:
    print("Usage: python inference.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
image = preprocess_image(img_path, target_size=(128, 128))  # returns (H, W, 3)
input_img = np.expand_dims(image, axis=0)

# --- Predict ---
pred_mask = model.predict(input_img)[0]  # shape: (128, 128, 1)
pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()

# --- Visualize ---
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap='gray')
plt.tight_layout()
plt.show()
