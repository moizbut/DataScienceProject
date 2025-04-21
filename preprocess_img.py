import os
import cv2
import numpy as np

# Paths (update with your actual paths)
DATA_ROOT = "data/data"
IMAGE_DIR1 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/test")
IMAGE_DIR2 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/UTRSet-Real/test/test")
OUTPUT_DIR1 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/data1")
OUTPUT_DIR2 = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/data/data2")
IMG_HEIGHT, IMG_WIDTH = 32, 128

# Create output directories
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)

def preprocess_image(img_path, output_path):
    """
    Preprocess an image: load in grayscale, resize to 128x32, save as .png.
    
    Args:
        img_path (str): Path to input image.
        output_path (str): Path to save preprocessed .png image.
    
    Returns:
        bool: True if successful, False if failed.
    """
    try:
        # Load in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load: {img_path}")
            return False
        # Resize to 128x32
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        # Save as grayscale PNG
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

# Process both datasets
for image_dir, output_dir in [(IMAGE_DIR1, OUTPUT_DIR1), (IMAGE_DIR2, OUTPUT_DIR2)]:
    print(f"Processing {image_dir} -> {output_dir}")
    if not os.path.exists(image_dir):
        print(f"Input directory not found: {image_dir}")
        continue
    processed, failed = 0, 0
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            output_name = img_name.split('.')[0] + '.png'  # Save as .png
            output_path = os.path.join(output_dir, output_name)
            if preprocess_image(img_path, output_path):
                processed += 1
            else:
                failed += 1
    print(f"Processed: {processed}, Failed: {failed}")