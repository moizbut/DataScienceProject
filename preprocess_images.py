import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Paths (update as needed)
DATA_ROOT = "data/data"
IMAGE_DIR1 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/test")
IMAGE_DIR2 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/UTRSet-Real/test/test")
OUTPUT_DIR1 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/data1")
OUTPUT_DIR2 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/data2")
IMG_HEIGHT, IMG_WIDTH = 32, 128

# Create output directories
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)

# Transform for normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # [0.0, 1.0] -> [-1.0, 1.0]
])

def preprocess_image(img_path, output_path, save_as_tensor=True):
    # Load in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load: {img_path}")
        return False
    # Resize to 128x32
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if save_as_tensor:
        # Convert to PIL Image
        img = Image.fromarray(img)
        # Apply transform
        img_tensor = transform(img)  # Shape: (1, 32, 128), values: [-1.0, 1.0]
        # Save as .npy
        np.save(output_path, img_tensor.numpy())
    else:
        # Save as grayscale PNG
        cv2.imwrite(output_path, img)
    return True

# Process both datasets
for image_dir, output_dir in [(IMAGE_DIR1, OUTPUT_DIR1), (IMAGE_DIR2, OUTPUT_DIR2)]:
    print(f"Processing {image_dir} -> {output_dir}")
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            output_name = img_name.split('.')[0] + ('.npy' if save_as_tensor else '.png')
            output_path = os.path.join(output_dir, output_name)
            preprocess_image(img_path, output_path, save_as_tensor=True)