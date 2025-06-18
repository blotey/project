import os
import numpy as np
from PIL import Image

def preprocess_image(img_path, target_size=(128, 128)):
    img = Image.open(img_path).resize(target_size)
    img = np.array(img)
    if img.ndim == 2:  # grayscale image
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img / 255.0

def preprocess_mask(mask_path, target_size=(128, 128)):
    mask = Image.open(mask_path).resize(target_size)
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]  # use one channel if 3D
    return (mask > 127).astype(np.float32)[..., np.newaxis]

def load_image_mask_pairs(image_dir, mask_dir, target_size=(128, 128)):
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        img = preprocess_image(img_path, target_size)
        mask = preprocess_mask(mask_path, target_size)
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)
